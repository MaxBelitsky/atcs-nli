# Learning sentence representations using a Natural Language Inference (NLI) task

The repository contains the code for the reproduction of the results from "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" paper by Conneau et al. (2018).

## Usage

### Installation
- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `. ./venv/bin/activate`
- Install the dependencies: `pip install -r requirements.txt`
- Install `SentEval` framework: `./scripts/install_senteval_pip.sh` (might not download the STS dataset on MacOS)
    - SentEval uses old code so the Python vesrion should be 3.9. A workaround can be found here: https://github.com/facebookresearch/SentEval/issues/89.
    - STS14 benchmark code in SentEval doesn't work with newer numpy versions, so this workaround needs to be appied in order to evaluate the models on STS14 dataset: https://github.com/facebookresearch/SentEval/issues/94.

### Model training
Example usage:
```
python -m src.train --model lstm
```
Acceptable model values are `lstm`, `bi-lstm`, `bi-lstm-pool`.
All arguments with their description can be viewed with `python -m src.train -h`.

### Model evaluation
Example usage:
```
python -m src.eval --model lstm --checkpoint_path models/lstm_2024_04_17_13_47.pt
```
Acceptable model values are `lstm`, `bi-lstm`, `bi-lstm-pool`, `mean`.
All arguments with their description can be viewed with `python -m src.eval -h`.

## Results

### Results on SNLI task

The accuracy is rounded to two decimal points.

| Model                   | Validation accuracy | Test accuracy |
|-------------------------|---------------------|---------------|
| LSTM                    |         81.07	    |      80.53    |
| BiLSTM                  |         80.55	    |      80.32    |
| BiLSTM with max pooling |         84.87       |      84.47    |


### Transfer results

| Model                   | MR | CR | SUBJ | MPQA | SST | TREC | MRPC | SICK-R | SICK-E | STS14 |
|-------------------------|----|----|------|------|-----|------|------|--------|--------|-------|
| GloVe BOW               |    |    |      |      |     |      |      |        |        |       |
| LSTM                    |    |    |      |      |     |      |      |        |        |       |
| BiLSTM                  |    |    |      |      |     |      |      |        |        |       |
| BiLSTM with max pooling |    |    |      |      |     |      |      |        |        |       |
