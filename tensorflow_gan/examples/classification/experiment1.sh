#!/usr/bin/env bash

#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_lr01 --optimizer RMSprop --base_learning_rate 0.1
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_lr001 --optimizer RMSprop --base_learning_rate 0.01
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_lr0001 --optimizer RMSprop --base_learning_rate 0.001

#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr01_e30 --optimizer RMSprop --base_learning_rate 0.1 --training_epochs 30
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr001_e30 --optimizer RMSprop --base_learning_rate 0.01 --training_epochs 30
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr0001_e30 --optimizer RMSprop --base_learning_rate 0.001 --training_epochs 30

#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr01_e50 --optimizer RMSprop --base_learning_rate 0.1 --training_epochs 50
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr001_e50 --optimizer RMSprop --base_learning_rate 0.01 --training_epochs 50
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr0001_e50 --optimizer RMSprop --base_learning_rate 0.001 --training_epochs 50

#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr01_e70 --optimizer RMSprop --base_learning_rate 0.1 --training_epochs 70
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr001_e70 --optimizer RMSprop --base_learning_rate 0.01 --training_epochs 70
#python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr0001_e70 --optimizer RMSprop --base_learning_rate 0.001 --training_epochs 70

python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr005_e50 --optimizer RMSprop --base_learning_rate 0.05 --training_epochs 50
python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr002_e50 --optimizer RMSprop --base_learning_rate 0.02 --training_epochs 50
python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr0005_e50 --optimizer RMSprop --base_learning_rate 0.005 --training_epochs 50
python -m tensorflow_gan.examples.classification.train_tfds --output_folder rmsp_nolrr_lr0002_e50 --optimizer RMSprop --base_learning_rate 0.002 --training_epochs 50
