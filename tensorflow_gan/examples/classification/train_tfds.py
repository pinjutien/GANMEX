#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd 
import tensorflow as tf

import tensorflow_datasets as tfds
import tensorflow.keras as keras
# import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow_gan.examples.classification.data_utils import *

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

ckpt_path = "./test_model/test_a2o/"
model_name = "apple2orange"
epochs= 1 if test_run else 50
batch_size=32
os.makedirs(ckpt_path, exist_ok=False)


# data_path = '/Users/ptien/tfds-download/horse2zebra/'
# df, data_path = get_tfds_df_path(data_path)
# train_generator, val_generator, train_size, val_size = get_generators_from_df_path(df, data_path, IMAGE_SIZE, 32)

train_generator, val_generator, train_size, val_size = get_generators_from_tfds('cycle_gan', IMAGE_SIZE, 32, test_run=test_run)

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

base_learning_rate = 0.0005

model.summary() # show model summary
model.compile(optimizer=tf.keras.optimizers.Adadelta(base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.save(ckpt_path + "base_model.h5".format(model_name=model_name))

for layer in model.layers:
    # things will break if the input_1 and the activation layers get renamed
    if not layer.name.startswith('input') and not layer.name.startswith('activation'):
        layer._name = 'Discriminator/custom_discriminator/' + layer.name

model.summary()

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

mdlckpt = ModelCheckpoint(ckpt_path + 'model-{epoch:03d}-{val_acc:03f}.h5', save_best_only=True, monitor='val_acc', mode='max')
callbacks = [earlystop, learning_rate_reduction, mdlckpt]

max_step = 1 if test_run else 150

history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=max(max_step, val_size//batch_size),
    steps_per_epoch=max(max_step, train_size//batch_size),
    callbacks=callbacks
)

model.save_weights(ckpt_path + "{model_name}_weights_keras.h5".format(model_name=model_name))
model.save(ckpt_path + "{model_name}_keras.h5".format(model_name=model_name))

tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=ckpt_path)

# # For spot-checking prediction numbers
# keras_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-2].output)
# import tensorflow_datasets as tfds
# ds = tfds.load('cycle_gan')
# examples_apples = list(tfds.as_numpy(ds['testA'].take(10)))
# import numpy as np
# inputs_apples = np.array([x['image'] for x in examples_apples]) / 255.0
# print(keras_model.predict(inputs_apples))
