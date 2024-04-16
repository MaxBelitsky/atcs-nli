import os
import logging
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from src.data import CustomCollator

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, dataset, tokenizer, args):
        self.model = model.to(args.device)
        self.train_data = dataset["train"]
        self.val_data = dataset["validation"]
        self.tokenizer = tokenizer
        self.args = args

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
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            self.val_data,
            collate_fn=collator,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
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
                }
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
                step_metrics = self.compute_metrics(model_output, labels)
                step_metrics["loss"] = loss.detach().item()
                self.log({"train": step_metrics})

            # Eval on validation set
            val_metrics = self.evaluate_model(self.model, self.val_dataloader)
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.save_model()
                # Shrink learning rate if the accuracy improves
                if epoch != 0:
                    self.scheduler_2.step()
            # Update learning rate
            self.scheduler.step()

            self.log({"val": val_metrics, "lr": self.scheduler.get_last_lr()})

    def evaluate_model(self, model, dataloader):
        model.eval()
        all_metrics = []
        for batch in dataloader:
            premises, hypotheses, labels = batch['premises'], batch['hypotheses'], batch['labels']
            with torch.no_grad():
                model_output = model(premises, hypotheses)
            metrics = self.compute_metrics(model_output, labels)
            all_metrics.append(metrics)
        
        return self._aggregate_metrics(all_metrics, dataloader)
    
    def _aggregate_metrics(self, all_metrics, dataloader):
        n_batches = len(dataloader)
        result = {}
        for batch_metrics in all_metrics:
            for metric, value in batch_metrics.items():
                result[metric] = result.get(metric, 0) + value
        
        for metric, value in result.items():
            result[metric] = result.get(metric, 0) / n_batches
        return result

    def compute_metrics(self, preds, target):
        # Compute accuracy
        accuracy = (preds.argmax(dim=-1) == target).float().mean()

        metrics = {"accuracy": accuracy}
        return metrics

    def save_model(self):
        os.makedirs(self.args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        model_file = os.path.join(self.args.output_dir, f"{self.args.model}_{timestamp}.pt")
        torch.save(self.model.state_dict(), model_file)

    def load_checkpoint_weights(self):
        # Check if file exists
        if not os.path.isfile(self.args.checkpoint_path):
            raise Exception(f"File {self.args.checkpoint_path} doesn't exist")

        # Load the weights
        self.model.load_state_dict(torch.load(self.args.checkpoint_path))
