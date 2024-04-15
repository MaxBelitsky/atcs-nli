import logging
import argparse

import torch
import senteval
import numpy as np

from src.utils import build_tokenizer, read_glove_embeddings
from src.models import MeanEmbedder

# PATH_TO_DATA = PATH_TO_SENTEVAL+'/data'
PATH_TO_DATA = 'pretrained'
# PATH_TO_VEC = 'pretrained/glove.6B.300d.txt'
# PATH_TO_VEC = 'pretrained/glove.840B.300d.txt'
# WORDS, VECTORS = read_glove_embeddings(PATH_TO_VEC)
WORDS, VECTORS = read_glove_embeddings("840B")

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


def prepare(params, samples):
    """
    In this example we are going to load Glove, 
    here you will initialize your model.
    remember to add what you model needs into the params dictionary
    """
    params.tokenizer = build_tokenizer(WORDS)
    params.embedder = MeanEmbedder(VECTORS)


def batcher(params, batch):
    """
    In this example we use the average of word embeddings as a sentence representation.
    Each batch consists of one vector for sentence.
    Here you can process each sentence of the batch, 
    or a complete batch (you may need masking for that).
    
    """
    # if a sentence is empty dot is set to be the only token
    # you can change it into NULL dependening in your model
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        # sentence = " ".join(sent)
        # tokens_ids = params.tokenizer(sentence, return_tensors="pt", padding=True)[
        #     "input_ids"
        # ]
        tokens_ids = torch.tensor(
            params.tokenizer.convert_tokens_to_ids(sent)
        ).unsqueeze(dim=0)

        with torch.no_grad():
            sentence_embeddings = params.embedder(tokens_ids)
        sentence_embeddings = sentence_embeddings.squeeze().numpy()
        embeddings.append(sentence_embeddings)
    # [batch size, embedding dimensionality]
    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
# this is the config for the NN classifier but we are going to use scikit-learn logistic regression with 10 kfold
# params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
#                                 'tenacity': 3, 'epoch_size': 2}


if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)

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
