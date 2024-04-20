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

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

logger = logging.getLogger(__name__)


def get_unique_tokens(sentences, normalizer, pre_tokenizer):
    unique_tokens = set()
    for sentence in sentences:
        normalized_sentence = normalizer.normalize_str(sentence)
        tokenized_sentence = pre_tokenizer.pre_tokenize_str(normalized_sentence)
        for token in tokenized_sentence:
            unique_tokens.add(token[0])
    logger.info(f"Found {len(unique_tokens)} unique tokens")
    return unique_tokens


def read_glove_embeddings(
    name: str = "840B",
    dim: int = 300,
    cache_dir=".vector_cache",
):
    glove = GloVe(name, dim, cache=cache_dir)
    return glove.itos, glove.vectors


def align_with_glove(word_freqs, words, vectors):
    aligned_words, aligned_vectors = [], []

    for word, vector in zip(words, vectors):
        if word in word_freqs:
            aligned_words.append(word)
            aligned_vectors.append(vector)
    # Add special tokens
    logger.info(f"Found {len(aligned_words)} tokens with embeddings")
    aligned_words = [PAD_TOKEN, UNK_TOKEN] + aligned_words
    aligned_vectors = torch.stack(aligned_vectors)
    aligned_vectors = torch.cat((torch.zeros(2, vectors.size(-1)), aligned_vectors))
    return aligned_words, aligned_vectors


def exctract_sentences(dataset):
    sentences = []
    for split in dataset:
        sentences.extend(list(set(dataset[split]["premise"] + dataset[split]["hypothesis"])))
    return sentences


def build_tokenizer(words, normalizer, pre_tokenizer) -> PreTrainedTokenizerFast:
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
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    # Save the tokenizer
    tokenizer.save("models/tokenizer.json")

    # Wrap the tokenizer to use in Transformers
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
    )
    return tokenizer


def align_vocab_and_build_tokenizer(sentences, embedding_cache_dir=".vector_cache"):
    # Create a lowercase normalizer
    normalizer = normalizers.Sequence([normalizers.Lowercase()])
    # Create a Whitespace pre-tokenizer – split text on whitespace
    pre_tokenizer = pre_tokenizers.Whitespace()

    # Build a vocabulary and align it with GloVe
    logger.info("Finding all unique tokens")
    unique_tokens = get_unique_tokens(sentences, normalizer, pre_tokenizer)
    logger.info("Aligning with GloVe vectors")
    words, vectors = read_glove_embeddings(cache_dir=embedding_cache_dir)
    words, vectors = align_with_glove(unique_tokens, words, vectors)

    # Create a tokenizer
    tokenizer = build_tokenizer(words, normalizer, pre_tokenizer)

    return tokenizer, vectors


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
