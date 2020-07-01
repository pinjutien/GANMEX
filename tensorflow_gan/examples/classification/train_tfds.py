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

test_run = True
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


def get_optimizer(args):
    module = importlib.import_module(tf.keras.optimizers)
    class_ = getattr(module, args.optimizer)
    optimizer = class_(learning_rate=args.learning_rate)
    return optimizer


def main(args):
    epochs= 1 if test_run else 50
    batch_size=32
    os.makedirs(args.output_path, exist_ok=False)

    train_generator, val_generator, train_size, val_size = get_generators_from_tfds(DATASET_NAME, IMAGE_SIZE, 32, test_run=test_run)

    # load model without classifier layers
    pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    # add new classifier layers
    flat1 = Flatten()(pre_model.output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(2, activation=None)(class1)
    output = Activation('softmax')(output)

    # define new model
    model = Model(inputs=pre_model.input, outputs=output)

    for layer in model.layers[:19]:
        layer.trainable = False

    print("Base model summary:")
    model.summary() # show model summary
    model.compile(optimizer=tf.keras.optimizers.Adadelta(args.base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.save(args.output_path + "base_model.h5".format(model_name=args.model_name))

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    callbacks = [earlystop, learning_rate_reduction]

    if args.save_checkpoints:
        mdlckpt = ModelCheckpoint(args.output_path + 'model-{epoch:03d}-{val_acc:03f}.h5', save_best_only=True, monitor='val_acc', mode='max')
        callbacks.append(mdlckpt)

    max_step = 1 if test_run else 150

    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=max(max_step, val_size//batch_size),
        steps_per_epoch=max(max_step, train_size//batch_size),
        callbacks=callbacks
    )

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

    test(args, model)


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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Optimizer
    parser.add_argument("--optimizer", default=None, type=str, required=True,
                        help="Currently supports: Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD")
    parser.add_argument("--base_learning_rate", default=0.01, type=float,
                        help="Starting learning rate of the optimizer")

    # Output
    parser.add_argument("--output_folder", default=None, type=str, required=True,
                        help="The output path will be ./test_model/<output_folder>/")
    parser.add_argument("--model_name", default='model', type=str,
                        help="File name of the final keras model")
    parser.add_argument("--save_checkpoints", action='store_true',
                        help="Save checkpoints")

    if test_run:
        test_args = [
            "--optimizer", "Adam",
            "--output_folder", "test_a2o_6/",
        ]
        args = parser.parse_args(test_args)
        print('Using the default test argument:')
    else:
        args = parser.parse_args()
        print('Parsed argument:')

    print(args)

    args.output_path = './test_model/%s/' % args.output_folder

    main(args)



