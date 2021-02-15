#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import math
import numpy as np
import importlib
import pandas as pd 
import tensorflow as tf

from numpy.random import seed
from tensorflow import set_random_seed

import tensorflow_datasets as tfds

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow_gan.examples.classification.data_utils import *
from tensorflow_gan.examples.classification.model_utils import *

from tensorflow_gan import custom_tfds


train_val_split = False
test_run = False
use_cpu = False
if (use_cpu):
    print("Use CPU only")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# IMAGE_WIDTH=256
# IMAGE_HEIGHT=256
# IMAGE_CHANNELS=3
# DATASET_NAME = 'cycle_gan'
# FILL_COLOR = False

IMAGE_WIDTH=28
IMAGE_HEIGHT=28
IMAGE_CHANNELS=1
DATASET_NAME = 'mnist'
FILL_COLOR = True

# IMAGE_WIDTH=32
# IMAGE_HEIGHT=32
# IMAGE_CHANNELS=3
# DATASET_NAME = 'svhn_cropped'
# FILL_COLOR = False

# IMAGE_WIDTH=32
# IMAGE_HEIGHT=32
# IMAGE_CHANNELS=3
# DATASET_NAME = 'cifar10'
# FILL_COLOR = False

# IMAGE_WIDTH=128
# IMAGE_HEIGHT=128
# IMAGE_CHANNELS=3
# DATASET_NAME = 'obj_scene_v2'
# FILL_COLOR = False

if FILL_COLOR:
    IMAGE_CHANNELS = 3

IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
BASE_PATH = './test_model/'

# Set random seed
seed(1)
set_random_seed(2)


def train(args, generator_struct):
    # epochs = 1 if test_run else 50
    batch_size = 32
    os.makedirs(args.output_path, exist_ok=False)
    with open(args.output_path + 'train_parameters.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=4)

    input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    if DATASET_NAME == 'mnist':
        model = get_mnist_model(input_shape)
    elif DATASET_NAME == 'svhn_cropped':
        model = get_svhn_model(input_shape)
    elif DATASET_NAME == 'cifar10':
        model = get_cifar10_model(input_shape)
    elif DATASET_NAME.startswith('cycle_gan'):
        model = get_cyclegan_model(input_shape, args.base_model)
    elif DATASET_NAME.startswith('obj_scene'):
        model = get_cyclegan_model(input_shape, args.base_model, num_classes=4)
    else:
        model = get_model(
            (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS),
            base_model_name=args.base_model,  # 'vgg16',
            additional_conv_layers=args.additional_conv_layers,
            global_average_pooling=args.global_average_pooling,
            global_max_pooling=args.global_max_pooling,
            conv_batch_normalization=args.conv_batch_normalization,
            conv_dropout=args.conv_dropout,
            dense_batch_normalization=args.dense_batch_normalization,
            dense_dropout=args.dense_dropout,
            dense_sizes=[int(x) for x in args.dense_sizes.split(',') if int(x)],
        )

    print("Base model summary:")
    model.summary()  # show model summary
    with open(args.output_path + 'report.txt', 'w') as fp:
        fp.write('\n\n')
        model.summary(print_fn=lambda x: fp.write(x + '\n'))
    model.save(args.output_path + "model_summary")
    model.compile(optimizer=get_optimizer(args),  # tf.keras.optimizers.Adadelta(args.base_learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.save(args.output_path + "base_model.h5")

    if 'val' in generator_struct:
        earlystop = EarlyStopping(patience=20) #10)
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                    patience=5, #2,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.00001)

        callbacks = [earlystop, learning_rate_reduction]

        if args.save_checkpoints:
            mdlckpt = ModelCheckpoint(args.output_path + 'model-{epoch:03d}-{val_acc:03f}.h5', save_best_only=True,
                                      monitor='val_acc', mode='max')
            callbacks.append(mdlckpt)

        history = model.fit_generator(
            generator_struct['train']['generator'],
            epochs=args.training_epochs,
            validation_data=generator_struct['val']['generator'],
            validation_steps=generator_struct['val']['steps'],  # max(max_step, val_size // batch_size),
            steps_per_epoch=generator_struct['train']['steps'],  # max(max_step, train_size // batch_size),
            callbacks=callbacks
        )
    else:
        # train_generator, train_size = get_generators_from_tfds(
        #     DATASET_NAME, IMAGE_SIZE, 32, test_run=test_run, train_val_split=train_val_split)
        history = model.fit_generator(
            generator_struct['train']['generator'],
            epochs=args.training_epochs,
            steps_per_epoch=generator_struct['train']['steps'],
        )

    print(history)

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


def quick_test(model, data_struct, batch_size):
    with open(args.output_path + 'metrics.txt', 'w') as fp:
        for key in ['train', 'val', 'test']:
            if key in data_struct:
                curr_x, curr_y = data_struct[key]
                [loss, acc] = model.evaluate(curr_x / 255.0, curr_y, batch_size=batch_size)
                print(key + '_loss:', loss)
                print(key + '_acc:', acc)
                fp.write('%s_loss: %.5f\n' % (key, loss))
                fp.write('%s_acc: %.5f\n' % (key, acc))


def test(args, model):
    # for spot check
    # keras_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)

    def evaluate(datasetA, datasetB):
        if test_run:
            examples_a = list(tfds.as_numpy(datasetA.take(10)))
            examples_b = list(tfds.as_numpy(datasetB.take(10)))
        else:
            examples_a = list(tfds.as_numpy(datasetA))
            examples_b = list(tfds.as_numpy(datasetB))
        inputs_a = np.array([np.array(Image.fromarray(a['image']).resize(IMAGE_SIZE)) for a in examples_a]) / 255.0
        inputs_b = np.array([np.array(Image.fromarray(a['image']).resize(IMAGE_SIZE)) for a in examples_b]) / 255.0

        preds_a = model.predict(inputs_a)
        preds_b = model.predict(inputs_b)

        accuracy = np.mean([int(x[0] > x[1]) for x in preds_a] + [int(x[1] > x[0]) for x in preds_b])
        cross_entropy = np.mean([-math.log(x[0]) for x in preds_a] + [-math.log(x[1]) for x in preds_b])

        return accuracy, cross_entropy

    ds = tfds.load(DATASET_NAME)
    # if test_run:
    #     examples_a = list(tfds.as_numpy(ds['testA'].take(10)))
    #     examples_b = list(tfds.as_numpy(ds['testB'].take(10)))
    # else:
    #     examples_a = list(tfds.as_numpy(ds['testA']))
    #     examples_b = list(tfds.as_numpy(ds['testB']))
    # inputs_a = np.array([x['image'] for x in examples_a]) / 255.0
    # inputs_b = np.array([x['image'] for x in examples_b]) / 255.0
    # preds_a = model.predict(inputs_a)
    # preds_b = model.predict(inputs_b)
    #
    # accuracy = np.mean([int(x[0] > x[1]) for x in preds_a] + [int(x[1] > x[0]) for x in preds_b])
    # cross_entropy = np.mean([-math.log(x[0]) for x in preds_a] + [-math.log(x[1]) for x in preds_b])

    train_accuracy, train_cross_entropy = evaluate(ds['trainA'], ds['trainB'])
    test_accuracy, test_cross_entropy = evaluate(ds['testA'], ds['testB'])

    print('Train Accuracy:', train_accuracy)
    print('Train Cross Entropy Loss:', train_cross_entropy)

    print('Test Accuracy:', test_accuracy)
    print('Test Cross Entropy Loss:', test_cross_entropy)

    filename = BASE_PATH + 'search_log.txt'
    if os.path.exists(filename):
        append_write = 'a'  # append if already exists
    else:
        append_write = 'w'  # make a new file if not

    with open(filename, append_write) as fp:
        fp.write('\n' + args.output_folder + '\n')
        fp.write('Train Accuracy: %f\n' % train_accuracy)
        fp.write('Train Cross Entropy Loss: %f\n' % train_cross_entropy)
        fp.write('Test Accuracy: %f\n' % test_accuracy)
        fp.write('Test Cross Entropy Loss: %f\n' % test_cross_entropy)


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
    parser.add_argument("--training_epochs", type=int, default=50,
                        help="Number of epochs in training")

    # Model params
    parser.add_argument("--base_model", type=str, default='VGG16',
                        help="Currently supports: VGG16, MobileNet, MobileNetV2")

    parser.add_argument("--additional_conv_layers", type=int, default=0,
                        help="Additional convoluational layers on top of the pre-trained model")
    parser.add_argument("--global_average_pooling", action='store_true',
                        help="Use a global average pooling layer")
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
            "--base_model", "MobileNet",
            "--optimizer", "Adadelta",
            "--base_learning_rate", "1.0",  # Adadelta should use a learning rate of 1.0
            "--output_folder", "testtest",
            "--training_epochs", "1",
            "--additional_conv_layers", "2",
            "--dense_sizes", "8096,1024",
            "--dense_dropout",
            "--dense_batch_normalization"
        ]
        args = parser.parse_args(test_args)
        print('Using the default test argument:')
    else:
        args = parser.parse_args()
        print('Parsed argument:')

    print(args)

    args.output_path = BASE_PATH + args.output_folder + '/'

    data_struct = get_datasets(DATASET_NAME, IMAGE_SIZE, test_run=test_run, train_val_split=train_val_split, fill_color=FILL_COLOR)
    generator_struct = get_generators(DATASET_NAME, data_struct, 32)

    generator_struct['val'] = generator_struct['test']  ############

    model = train(args, generator_struct)
    save_model(args, model)
    quick_test(model, data_struct, 32)

