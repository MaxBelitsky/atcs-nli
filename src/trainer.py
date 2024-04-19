import os
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from src.data import CustomCollator
from src.utils import compute_metrics, evaluate_model, load_checkpoint_weights

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, dataset, tokenizer, args):
        self.model = model.to(args.device)
        self.train_data = dataset["train"]
        self.val_data = dataset["validation"]
        if "test" in dataset:
            self.test_data = dataset["test"]
        self.tokenizer = tokenizer
        self.args = args
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.model_file = os.path.join(self.args.output_dir, f"{self.args.model}_{self.timestamp}.pt")
        self.embedding_model_file = os.path.join(
            self.args.output_dir, f"{self.args.model}_embedder.pt"
        )
        logger.info(f"Training run initialized at {self.timestamp}")

        # Load weights from a checkpoints if the path is provided
        if self.args.checkpoint_path:
            self.load_checkpoint_weights()

        # Create dataloaders
        collator = CustomCollator(tokenizer, device=args.device)
        self.train_dataloader = DataLoader(
            self.train_data,
            collate_fn=collator,
            batch_size=args.batch_size,
            shuffle=True,
        )
        self.val_dataloader = DataLoader(
            self.val_data,
            collate_fn=collator,
            batch_size=args.batch_size,
            shuffle=True,
        )
        if "test" in dataset:
            self.test_dataloader = DataLoader(
                self.test_data,
                collate_fn=collator,
                batch_size=args.batch_size,
                shuffle=True,
            )

        # Initialize the optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=args.lr_decay)
        self.scheduler_2 = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=args.lr_shrink)

        # Initialize a loss function
        self.loss_function = torch.nn.CrossEntropyLoss()

        if args.use_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="uva-acts",

                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "architecture": args.model,
                    "epochs": args.epochs,
                },
                name=f"{self.args.model}_{self.timestamp}",
            )

    def log(self, metrics):
        if self.args.use_wandb:
            wandb.log(metrics)

        if self.args.use_tensorboard:
            pass

    def train_model(self):
        best_val_accuracy = 0
        for epoch in range(self.args.epochs):
            logger.info(f"Epoch: {epoch} | Learning rate: {self.scheduler.get_last_lr()}")
            self.model.train()
            for batch in self.train_dataloader:
                premises, hypotheses, labels = batch['premises'], batch['hypotheses'], batch['labels']
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                model_output = self.model(premises, hypotheses)

                # Compute loss
                loss = self.loss_function(model_output, labels)

                # Backward pass and weights update
                loss.backward()
                self.optimizer.step()

                # Compute and log training metrics
                step_metrics = compute_metrics(model_output, labels)
                step_metrics["loss"] = loss.detach().item()
                self.log({"train": step_metrics})

            # Eval on validation set
            val_metrics = self.evaluate_model(split="val")
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model()
            elif epoch > 0:
                # Shrink learning rate if the accuracy decreases
                logger.info("Shrinking learning rate")
                self.scheduler_2.step()

            # Update learning rate with the main scheduler
            if epoch > 0:
                self.scheduler.step()

            self.log({"val": val_metrics, "lr": self.scheduler.get_last_lr()[0]})

            # Stop the training if the learning rate is lower than the minimum
            if self.scheduler.get_last_lr()[0] < self.args.min_lr:
                logger.info(f"Stopping training at epoch: {epoch}. Current lr: {self.scheduler.get_last_lr()}")
                break

    def evaluate_model(self, split="val"):
        self.model.eval()

        if split == "val":
            dataloader = self.val_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        return evaluate_model(self.model, dataloader)

    def save_model(self, skip_glove=True):
        os.makedirs(self.args.output_dir, exist_ok=True)
        logger.info(f"Saving the best model to: {self.model_file}")

        state_dict = self.model.state_dict()
        if skip_glove:
            del state_dict['embedder.embedding.weight']
        torch.save(state_dict, self.model_file)

    def save_embedding_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        logger.info(f"Saving the embedding model to: {self.embedding_model_file}")
        torch.save(self.model.embedder.lstm.state_dict(), self.embedding_model_file)

    def load_checkpoint_weights(self, checkpoint_path=None, skip_glove=True):
        checkpoint_path = checkpoint_path or self.args.checkpoint_path
        # Check if file exists
        if not os.path.isfile(checkpoint_path):
            raise Exception(f"File {checkpoint_path} doesn't exist")
        # Load the weights
        load_checkpoint_weights(self.model, checkpoint_path, self.args.device, skip_glove)
