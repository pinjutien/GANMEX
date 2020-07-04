#!/usr/bin/env bash

python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_lr01 --optimizer RMSprop --base_learning_rate 0.1
python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_lr001 --optimizer RMSprop --base_learning_rate 0.01
python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_lr0001 --optimizer RMSprop --base_learning_rate 0.001
