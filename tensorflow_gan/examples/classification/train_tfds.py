#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import math
import numpy as np
import importlib
import pandas as pd 
import tensorflow as tf

import tensorflow_datasets as tfds

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow_gan.examples.classification.data_utils import *
from tensorflow_gan.examples.classification.model_utils import get_model

test_run = False
use_cpu = False
if (use_cpu):
    print("Use CPU only")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

IMAGE_WIDTH=256
IMAGE_HEIGHT=256
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3
DATASET_NAME = 'cycle_gan'

BASE_PATH = './test_model/'


def get_optimizer(args):
    module = importlib.import_module(tf.keras.optimizers)
    class_ = getattr(module, args.optimizer)
    optimizer = class_(learning_rate=args.learning_rate)
    return optimizer


def train(args):
    epochs = 1 if test_run else 50
    batch_size = 32
    os.makedirs(args.output_path, exist_ok=False)

    train_generator, val_generator, train_size, val_size = get_generators_from_tfds(DATASET_NAME, IMAGE_SIZE, 32,
                                                                                    test_run=test_run)

    model = get_model(
        (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
        base_model='vgg16',
        additional_conv_layers=args.additional_conv_layers,
        global_average_pooling=args.global_average_pooling,
        global_max_pooling=args.global_max_pooling,
        conv_batch_normalization=args.conv_batch_normalization,
        conv_dropout=args.conv_dropout,
        dense_batch_normalization=args.dense_batch_normalization,
        dense_dropout=args.dense_dropout,
        dense_sizes=[int(x) for x in args.dense_sizes.split(',')],
    )

    print("Base model summary:")
    model.summary()  # show model summary
    model.save(args.output_path + "model_summary")
    model.compile(optimizer=tf.keras.optimizers.Adadelta(args.base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.save(args.output_path + "base_model.h5")

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    if args.save_checkpoints:
        mdlckpt = ModelCheckpoint(args.output_path + 'model-{epoch:03d}-{val_acc:03f}.h5', save_best_only=True,
                                  monitor='val_acc', mode='max')
        callbacks.append(mdlckpt)

    max_step = 1 if test_run else 150

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=max(max_step, val_size // batch_size),
        steps_per_epoch=max(max_step, train_size // batch_size),
        callbacks=callbacks
    )

    return model


def save_model(args, model):
    model.save_weights(args.output_path + "{model_name}_weights.h5".format(model_name=args.model_name))
    model.save(args.output_path + "{model_name}_keras.h5".format(model_name=args.model_name))

    # Renaming the layer names will mess up the keras model, so need to do it after the keras models are saved
    for layer in model.layers:
        # things will break if the input_1 and the activation layers get renamed
        if not layer.name.startswith('input') and not layer.name.startswith('activation'):
            layer._name = 'Discriminator/custom_discriminator/' + layer.name

    print("Estimator model summary:")
    model.summary()

    tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=args.output_path)


def test(args, model):
    # for spot check
    # keras_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    ds = tfds.load(DATASET_NAME)
    examples_a = list(tfds.as_numpy(ds['testA'].take(10)))
    examples_b = list(tfds.as_numpy(ds['testB'].take(10)))
    inputs_a = np.array([x['image'] for x in examples_a]) / 255.0
    inputs_b = np.array([x['image'] for x in examples_b]) / 255.0
    preds_a = model.predict(inputs_a)
    preds_b = model.predict(inputs_b)

    accuracy = np.mean([int(x[0] > x[1]) for x in preds_a] + [int(x[1] > x[0]) for x in preds_b])
    cross_entropy = np.mean([-math.log(x[0]) for x in preds_a] + [-math.log(x[1]) for x in preds_b])
    print('Accuracy:', accuracy)
    print('Cross Entropy Loss:', cross_entropy)

    filename = BASE_PATH + 'search_log.txt'
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(filename, append_write) as fp:
        fp.write('\n' + args.output_folder + '\n')
        fp.write('Accuracy: %f\n' % accuracy)
        fp.write('Cross Entropy Loss: %f\n' % cross_entropy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="The output path will be ./test_model/<output_folder>/")
    parser.add_argument("--model_name", default='model', type=str,
                        help="File name of the final keras model")
    parser.add_argument("--save_checkpoints", action='store_true',
                        help="Save checkpoints")

    # Optimizer
    parser.add_argument("--optimizer", default=None, type=str, required=True,
                        help="Currently supports: Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD")
    parser.add_argument("--base_learning_rate", default=0.01, type=float,
                        help="Starting learning rate of the optimizer")

    # Model params
    parser.add_argument("--additional_conv_layers", type=int, default=0,
                        help="Additional convoluational layers on top of the pre-trained model")
    parser.add_argument("--global_average_pooling", action='store_true',
                        help="Use a global average pooling layer")
    parser.add_argument("--global_max_pooling", action='store_true',
                        help="Use a global average max layer")
    parser.add_argument("--conv_batch_normalization", action='store_true',
                        help="Add a batch normalization layer for each convoluational layer")
    parser.add_argument("--conv_dropout", action='store_true',
                        help="Add a dropout layer for each convoluational layer")
    parser.add_argument("--dense_batch_normalization", action='store_true',
                        help="Add a batch normalization layer for each dense layer")
    parser.add_argument("--dense_dropout", action='store_true',
                        help="Add a dropout layer for each dense layer")
    parser.add_argument("--dense_sizes", default='1024', type=str,
                        help="Sizes of dense layers. Should be a string consisting of numbers separated by commas")

    if test_run:
        test_args = [
            "--optimizer", "RMSprop",
            "--output_folder", "a2o_rmsp_t1",
            "--global_max_pooling",
            "--additional_conv_layers", "2",
            "--dense_sizes", "256"
        ]
        args = parser.parse_args(test_args)
        print('Using the default test argument:')
    else:
        args = parser.parse_args()
        print('Parsed argument:')

    print(args)

    args.output_path = BASE_PATH + args.output_folder + '/'

    model = train(args)
    test(args, model)
    save_model(args, model)
