import os
import logging

import torch
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
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
            dataset["train"],
            collate_fn=collator,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.val_dataloader = DataLoader(
            dataset["validation"],
            collate_fn=collator,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
        )

        # Initialize the optimizer
        # self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.99)

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
        for epoch in tqdm(range(self.args.epochs)):
            logger.info(f"Epoch: {epoch} | Learning rate: {self.scheduler.get_last_lr()}")
            print(f"Epoch: {epoch} | Learning rate: {self.scheduler.get_last_lr()}")
            self.model.train()
            for batch in self.train_dataloader:
                premises, hypotheses, labels = batch['premises'], batch['hypotheses'], batch['labels']
                # Reset gradients
                self.optimizer.zero_grad()

                # Forward pass
                model_output = self.model(premises['input_ids'], hypotheses['input_ids'])

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
                # TODO: save checkpoint
                # TODO: scheduler step
            self.log({"val": val_metrics})

            # Update learning rate
            self.scheduler.step()

    def evaluate_model(self, model, dataloader):
        model.eval()
        all_metrics = []
        for batch in dataloader:
            premises, hypotheses, labels = batch['premises'], batch['hypotheses'], batch['labels']
            with torch.no_grad():
                model_output = model(premises['input_ids'], hypotheses['input_ids'])
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
        model_file = os.path.join(self.args.output_dir, f"{self.args.model}.pt")
        torch.save(self.model.state_dict(), model_file)

    def load_checkpoint_weights(self):
        # Check if file exists
        if not os.path.isfile(self.args.checkpoint_path):
            raise Exception(f"File {self.args.checkpoint_path} doesn't exist")

        # Load the weights
        self.model.load_state_dict(torch.load(self.args.checkpoint_path))
