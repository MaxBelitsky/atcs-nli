from typing import Dict, List
import logging

import torch
from datasets import load_dataset
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer,
)
from transformers import PreTrainedTokenizerFast

DATASET_PATH = "stanfordnlp/snli"
DEFAULT_GLOVE_PATH = "pretrained/glove.840B.300d.txt"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

logger = logging.getLogger(__name__)


def read_glove_embeddings(file_path: str = None, dim: int = 300) -> Dict[str, list]:
    """
    Reads the GloVe embeddings from the provided file path.
    If the path not provided, glove.840B.300d is loaded.

    Args:
        file_path (str): path to the file with GloVe embeddings
        dim (int): the dimensionality of vectors
    Returns:
        embeddings (dict): a dictionary mapping the words to their
            embeddings
    """

    file_path = file_path or DEFAULT_GLOVE_PATH
    logger.info(f"Reading embeddings from {file_path}")

    words = [PAD_TOKEN, UNK_TOKEN]
    embeddings = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split(" ")
            word = values[0]
            try:
                vector = [float(val) for val in values[1:]]
                if len(vector) != dim:
                    continue
            except:
                # Skip ". . ." and "at name@domain.com"
                continue
            words.append(word)
            embeddings.append(vector)
    # Insert embeddings for special tokens
    embedding_dim = len(vector)
    embeddings.insert(0, [0]*embedding_dim)
    embeddings.insert(1, [0]*embedding_dim)

    # Convert embeddings to torch
    embeddings = torch.tensor(embeddings)
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


def get_dataset():
    return load_dataset(DATASET_PATH)
