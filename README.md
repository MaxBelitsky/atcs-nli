# Learning sentence representations using a Natural Language Inference (NLI) task

The repository contains the code for the reproduction of the results from "Supervised Learning of Universal Sentence Representations from Natural Language Inference Data" paper by Conneau et al. (2018).

## Usage

### Installation
- Create a virtual environment: `python -m venv venv`
- Activate the virtual environment: `. ./venv/bin/activate`
- Install the dependencies: `pip install -r requirements.txt`
- Install `SentEval` framework: `./scripts/install_senteval.sh` (might not download the STS dataset on MacOS)
- SentEval troublshooting: SentEval uses old code so the Python versions =>3.10 can lead to errors:
    - `ValueError: Function has keyword-only parameters or annotations, use inspect.signature() API which can support them`: A fix/workaround can be found here: https://github.com/facebookresearch/SentEval/issues/89.
    - `ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (750,) + inhomogeneous part.` on STS14 benchmark. STS14 benchmark code in SentEval doesn't work with newer numpy versions. A fix/workaround can be found here: https://github.com/facebookresearch/SentEval/issues/94.

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

Training logs can be found in the [training report](https://wandb.ai/mbelitsky/uva-acts/reports/Untitled-Report--Vmlldzo3NjI3MTQ5?accessToken=jbxp2lji6h7s6nndksome0ujvp82s6dbc9pjujp5gjojfj7gk7jxr0ko59luzwi4).

The pre-trained models checkpoints can be found [here](https://drive.google.com/drive/folders/18yYWQJ3VB4N7PvhdZDPIs4sUC-TOHe_W?usp=sharing).

### Results on SNLI task

The results on the SNLI task. The accuracy is rounded to two decimal points.

| Model                   | Validation accuracy | Test accuracy |
|-------------------------|---------------------|---------------|
| LSTM                    |         81.07	    |      80.53    |
| BiLSTM                  |         80.55	    |      80.32    |
| BiLSTM with max pooling |         84.87       |      84.47    |

### Averaged results on transfer tasks

Following the methodology of Conneau et al. (2018), ”micro” and ”macro” averages of development set (dev) results on transfer tasks whose metrics is accuracy. The accuracy is rounded to two decimal points.

| Model                   | Micro | Macro |
|-------------------------|-------|-------|
| Mean                    | 77.24 | 74.19 |
| LSTM                    | 79.75 | 78.94 |
| BiLSTM                  | 82.63 | 81.7  |
| BiLSTM with max pooling | 84.25 | 83.27 |


### Results on specific transfer results
| Model                   | MR    | CR    | SUBJ  | MPQA  | SST   | TREC | MRPC        | SICK-R | SICK-E | STS14     |
|-------------------------|-------|-------|-------|-------|-------|------|-------------|--------|--------|-----------|
| GloVe BOW               |       |       |       |       |       |      |             |        |        |           |
| LSTM                    | 73.86 | 77.69 | 86.38 | 87.69 | 77.98 | 75.4 | 73.04/81.39 | 0.8627 | 84.33  | 0.14/0.32 |
| BiLSTM                  | 74.6  | 79.08 | 89.33 | 88.06 | 79.41 | 87.8 | 73.57/82.12 | 0.8719 | 84.96  | 0.30/0.30 |
| BiLSTM with max pooling | 77.89 | 81.22 | 91.87 | 88.15 | 83.03 | 87.4 | 75.07/83.28 | 0.8824 | 85.06  | 0.69/0.67 |
