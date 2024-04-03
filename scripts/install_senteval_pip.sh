#!/bin/sh

# Install the library
pip install git+https://github.com/facebookresearch/SentEval.git

# Download the data
./scripts/download_data.sh
