from typing import List
import logging

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


def read_glove_embeddings(name: str = "840B", dim: int = 300):
    glove = GloVe(name, dim)
    words = [PAD_TOKEN, UNK_TOKEN] + glove.itos
    embeddings = torch.cat((torch.zeros(2, dim), glove.vectors))
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
