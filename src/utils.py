from typing import List, Optional
import logging
from collections import OrderedDict

import numpy as np
import torch
from torchtext.vocab import GloVe
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

DEFAULT_GLOVE_PATH = "pretrained/glove.840B.300d.txt"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

logger = logging.getLogger(__name__)


def read_glove_embeddings(
    name: str = "840B",
    dim: int = 300,
    topk: Optional[int] = None,
    cache_dir: Optional[str] = ".vector_cache",
):
    glove = GloVe(name, dim, cache=cache_dir)
    words = [PAD_TOKEN, UNK_TOKEN] + glove.itos
    embeddings = torch.cat((torch.zeros(2, dim), glove.vectors))
    if topk:
        words = words[:topk+2]
        embeddings = embeddings[:topk+2]
    return words, embeddings


def build_tokenizer(words: List[str]) -> PreTrainedTokenizerFast:
    """
    Creates a tokenizer.

    Args:
        embedding_dict (dict): a dictionary, where keys are words and values are embeddings.
    Returns:
        tokenizer (PreTrainedTokenizerFast): a tokenizer.
    """
    vocab = {word: id for id, word in enumerate(words)}

    # Use most simple tokenizer model based on mapping tokens to their corresponding id
    tokenizer_model = models.WordLevel(vocab=vocab, unk_token=UNK_TOKEN)

    # Initialize the tokenizer
    tokenizer = Tokenizer(tokenizer_model)

    # Add lowercase normalizer
    tokenizer.normalizer = normalizers.Sequence([normalizers.Lowercase()])

    # Add a Whitespace pre-tokenizer – split text on whitespace
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace() # can use Whitespace() for whitespace + punctuation

    # Wrap the tokenizer to use in Transformers
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
    )
    return tokenizer


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except:
        pass


def set_device():
    """
    Function for setting the device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    try:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
    except:
        device = torch.device('cpu')
    return device


def evaluate_model(model, dataloader):
    all_metrics = []

    for batch in dataloader:
        premises, hypotheses, labels = batch['premises'], batch['hypotheses'], batch['labels']
        with torch.no_grad():
            model_output = model(premises, hypotheses)
        metrics = compute_metrics(model_output, labels)
        all_metrics.append(metrics)
    
    return _aggregate_metrics(all_metrics, dataloader)


def _aggregate_metrics(all_metrics, dataloader):
    n_batches = len(dataloader)
    result = {}
    for batch_metrics in all_metrics:
        for metric, value in batch_metrics.items():
            result[metric] = result.get(metric, 0) + value
    
    for metric, value in result.items():
        result[metric] = result.get(metric, 0) / n_batches
    return result


def compute_metrics(preds, target):
    # Compute accuracy
    accuracy = (preds.argmax(dim=-1) == target).float().mean()

    metrics = {"accuracy": accuracy}
    return metrics


def load_checkpoint_weights(model, checkpoint_path, device, skip_glove, embedder_only=False):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if embedder_only:
        state_dict = select_embedder_keys(state_dict)

    if skip_glove:
        incompatible_keys = model.load_state_dict(state_dict=state_dict, strict=False)
        # Check if loading went as expected
        if incompatible_keys.unexpected_keys:
            raise Exception(
                f"The state dict has unexpected keys {incompatible_keys.unexpected_keys}"
            )
    else:
        model.load_state_dict(state_dict)


def select_embedder_keys(state_dict):
    new_state_dict = OrderedDict()
    state_dict.keys()

    for k, v in state_dict.items():
        if k.startswith('embedder'):
            new_key = ".".join(k.split('.')[1:])
            new_state_dict[new_key] = v
    return new_state_dict
