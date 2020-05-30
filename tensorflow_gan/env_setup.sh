#!/usr/bin/env bash

conda create --name tfgan_tf1 --clone tensorflow_p36
source activate tfgan_tf1
pip install tensorflow_probability==0.8.0
pip install tensorflow_datasets
pip install tensorflow_hub
conda deactivate