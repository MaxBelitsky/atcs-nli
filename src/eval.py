import logging
import argparse

import torch
import senteval

from src.constants import AvailableEmbedders
from src.utils import (
    build_tokenizer,
    read_glove_embeddings,
    load_checkpoint_weights,
    set_device,
    set_seed,
)
from src.models import MeanEmbedder, LSTMEmbedder, BiLSTMEmbedder, BiLSTMPooledEmbedder

PATH_TO_DATA = 'pretrained'
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def batcher(params, batch):
    """
    Exctracts embeddings from a batch of examples.
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if len(sent) > 0 else ['.'] for sent in batch]
    embeddings = []

    # Process batch
    tokenized_inputs = params.tokenizer(
        batch, return_tensors="pt", is_split_into_words=True, padding=True
    )
    tokenized_inputs["length"] = tokenized_inputs['attention_mask'].sum(dim=1).cpu()
    with torch.no_grad():
        sentence_embeddings = params.embedder(tokenized_inputs)
    embeddings = sentence_embeddings.squeeze().numpy()
    return embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training sentence embedding models")

    ### MODEL ###
    parser.add_argument(
        "--model",
        type=str,
        help="The model variant to evaluate",
        required=True,
        choices=AvailableEmbedders.values() + ["mean"]
    )
    parser.add_argument(
        "--lstm_n_hidden",
        type=int,
        default=2048,
        help="The hidden dimension of the LSTM"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="The batch size"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="The model checkpoint to evaluate"
    )
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default=".vector_cache",
        help="The directory to save the GloVe embeddings"
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="The directory to save the dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1111,
        help="The random seed for reproducibility"
        )
    parser.add_argument(
        "--device", type=str, default=None, help="The device to use for evaluation"
    )

    args = parser.parse_args()

    # Set seed and device
    set_seed(args.seed)
    args.device = args.device or set_device()

    # Load GloVe embeddings
    words, vectors = read_glove_embeddings(cache_dir=args.embedding_cache_dir)

    # Load/build the tokenizer
    tokenizer = build_tokenizer(words)

    # Initialize the embedding model and auxiluary model
    if args.model == AvailableEmbedders.LSTM:
        embedder = LSTMEmbedder(vectors, n_hidden=args.lstm_n_hidden)
    elif args.model == AvailableEmbedders.BI_LSTM:
        embedder = BiLSTMEmbedder(vectors, n_hidden=args.lstm_n_hidden)
    elif args.model == AvailableEmbedders.BI_LSTM_POOL:
        embedder = BiLSTMPooledEmbedder(vectors, n_hidden=args.lstm_n_hidden)
    elif args.model == "mean":
        embedder = MeanEmbedder(vectors)

    # Load the pretrained weights
    if args.model != "mean":
        load_checkpoint_weights(
            embedder, args.checkpoint_path, args.device, skip_glove=True, embedder_only=True
        )

    # Set params for SentEval
    params_senteval = {
        "task_path": PATH_TO_DATA,
        "usepytorch": False,
        "kfold": 10,
        "embedder": embedder,
        "tokenizer": tokenizer,
        "seed": args.seed,
        "batch_size": args.batch_size
    }

    se = senteval.engine.SE(params_senteval, batcher)

    transfer_tasks = [
        "MR",
        "CR",
        "MPQA",
        "SUBJ",
        "SST2",
        "TREC",
        "MRPC",
        "SICKEntailment",
    ]  # TODO: do smth about STS14 breaking
    # senteval prints the results and returns a dictionary with the scores
    results = se.eval(transfer_tasks)
    print(results) # TODO: save the results to JSON
