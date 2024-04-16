import torch
from datasets import load_dataset

DATASET_PATH = "stanfordnlp/snli"


class CustomCollator:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, batch):
        premises, hypotheses, labels = [], [], []

        for example in batch:
            premises.append(example["premise"])
            hypotheses.append(example["hypothesis"])
            labels.append(example["label"])

        # Tokenize sequences
        premises = self.tokenizer(
            premises,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        hypotheses = self.tokenizer(
            hypotheses,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        premises['length'] = premises["attention_mask"].sum(dim=1).to("cpu")
        hypotheses['length'] = hypotheses["attention_mask"].sum(dim=1).to("cpu")

        return {
            "premises": premises,
            "hypotheses": hypotheses,
            "labels": torch.tensor(labels).to(self.device)
        }


def get_dataset(file_path=None):
    file_path = file_path or DATASET_PATH
    # TODO: sort dataset by length for efficiency
    return load_dataset(file_path)
