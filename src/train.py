import argparse

from src.models import LSTMEmbedder, BiLSTMEmbedder, BiLSTMPooledEmbedder, SentenceClassificationModel
from src.utils import read_glove_embeddings, build_tokenizer, get_dataset
from src.constants import AvailableEmbedders


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

    args = parser.parse_args()

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

    # Set up logging

    # Define training arguments
    training_args = None

    # Initialize trainer
    trainer = None

    trainer.train()
    trainer.save_model()
