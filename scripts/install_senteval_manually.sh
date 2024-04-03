#!/bin/sh

# Clone the library
git clone https://github.com/facebookresearch/SentEval.git
# pip install git+https://github.com/facebookresearch/SentEval.git

# Install the library
cd SentEval/
python setup.py install

# Download the data
cd data/downstream/
./get_transfer_data.bash

# Return to the root
cd ../../..
