import argparse

from src.models import LSTMEmbedder, BiLSTMEmbedder, BiLSTMPooledEmbedder, SentenceClassificationModel
from src.utils import read_glove_embeddings, build_tokenizer
from src.data import get_dataset
from src.constants import AvailableEmbedders
from src.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training sentence embedding models")

    parser.add_argument(
        "--model",
        type=str,
        help="The model variant to train",
        required=True,
        choices=AvailableEmbedders.values()
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=512,
        help="The hidden dimension of the MLP head"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="The learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="The directory to save models to"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="The model checkpoint to continue training from"
    )
    # TODO: add other arguments

    args = parser.parse_args()

    # TODO: set seed

    # Load GloVe embeddings
    words, vectors = read_glove_embeddings()

    # Load/build the tokenizer
    tokenizer = build_tokenizer(words)

    # Initialize the embedding model and auxiluary model
    if args.model == AvailableEmbedders.LSTM:
        embedder = LSTMEmbedder(vectors)
    elif args.model == AvailableEmbedders.BI_LSTM:
        embedder = BiLSTMEmbedder(vectors)
    elif args.model == AvailableEmbedders.BI_LSTM_POOL:
        embedder = BiLSTMPooledEmbedder(vectors)

    model = SentenceClassificationModel(embedder, args.mlp_hidden_dim, 3)

    # Initialize the dataset
    dataset = get_dataset()

    # Train the model
    trainer = Trainer(model, dataset, tokenizer, args)
    trainer.train_model()
