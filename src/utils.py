from typing import Dict
import logging

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


def read_glove_embeddings(file_path: str = None) -> Dict[str, list]:
    """
    Reads the GloVe embeddings from the provided file path.
    If the path not provided, glove.840B.300d is loaded.

    Args:
        file_path (str): path to the file with GloVe embeddings
    Returns:
        embeddings (dict): a dictionary mapping the words to their
            embeddings
    """

    file_path = file_path or DEFAULT_GLOVE_PATH
    logger.info(f"Reading embeddings from {file_path}")

    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            try:
                vector = [float(val) for val in values[1:]]
            except:
                # Skip ". . ." and "at name@domain.com"
                continue
            embeddings[word] = vector
    return embeddings


def build_tokenizer(embedding_dict: Dict[str, list]) -> PreTrainedTokenizerFast:
    """
    Creates a tokenizer.

    Args:
        embedding_dict (dict): a dictionary, where keys are words and values are embeddings.
    Returns:
        tokenizer (PreTrainedTokenizerFast): a tokenizer.
    """
    vocab = {word: id for word, id in zip(embedding_dict.keys(), range(2, len(embedding_dict.keys())+2))}
    # vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    vocab[PAD_TOKEN] = 0
    vocab[UNK_TOKEN] = 1

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


def get_data():
    pass
