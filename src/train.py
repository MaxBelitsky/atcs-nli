import argparse
import logging

from src.models import LSTMEmbedder, BiLSTMEmbedder, BiLSTMPooledEmbedder, SentenceClassificationModel
from src.utils import read_glove_embeddings, build_tokenizer, set_device, set_seed
from src.data import get_dataset
from src.constants import AvailableEmbedders
from src.trainer import Trainer

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training sentence embedding models")

    ### MODEL ###
    parser.add_argument(
        "--model",
        type=str,
        help="The model variant to train",
        required=True,
        choices=AvailableEmbedders.values()
    )

    ### HYPERPARAMETERS ###
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=512,
        help="The hidden dimension of the MLP head"
    )
    parser.add_argument(
        "--lstm_n_hidden",
        type=int,
        default=2048,
        help="The hidden dimension of the LSTM"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="The number of epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="The learning rate"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-5,
        help="The minimum leraning rate (used for stopping condition)"
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.99,
        help="Learning rate decay factor for the lr scheduler"
    )
    parser.add_argument(
        "--lr_shrink",
        type=float,
        default=0.2,
        help="Learning rate shrink factor for the second lr scheduler"
    )

    ### SYSTEM ###
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="The directory to save models to"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="The model checkpoint to continue training from"
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
        default=1234,
        help="The random seed for reproducibility"
        )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="The device to use for training"
        )
    
    ### LOGGING ###
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to log to W&B"
        )
    parser.add_argument(
        "--use_tensorboard",
        action=argparse.BooleanOptionalAction,
        help="Indicates whether to log to Tensorboard"
        )
    

    args = parser.parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    args.device = args.device or set_device()

    logger.info(f"Beginning training with arguments: {args}")

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

    model = SentenceClassificationModel(embedder, args.mlp_hidden_dim, 3)

    # Initialize the dataset
    dataset = get_dataset(cache_dir=args.data_cache_dir)

    # Train the model
    trainer = Trainer(model, dataset, tokenizer, args)
    trainer.train_model()
    
    # Eval on the test set
    logger.info("Loading the best model")
    trainer.load_checkpoint_weights(trainer.model_file)
    test_metrics = trainer.evaluate_model(split="test")
    trainer.log({"test": test_metrics})
