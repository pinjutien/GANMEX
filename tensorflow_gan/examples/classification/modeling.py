#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow.keras as keras
# import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# from keras.applications import VGG16
# from keras.models import Model
# from keras.layers import Dense
# from keras.layers import Flatten

use_cpu = False
if (use_cpu):
    print("Use CPU only")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

data_path = '/home/ec2-user/apple2orange/'
ckpt_path = "./test_model/"
model_name = "apple2orange"
epochs=100
batch_size=32
os.makedirs(ckpt_path, exist_ok=True)

def get_data(data_path):
    filenames = []
    categories = []
    label_map = {
        'trainA': 'apple',
        'trainB': 'orange',
    }
    for folder in os.listdir(data_path):
        if folder in label_map.keys():
            for file in os.listdir(data_path + '/' + folder):
                categories.append(label_map[folder])
                filenames.append(folder + '/' + file)

    table = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return table, data_path

df, data_path = get_data(data_path)


IMAGE_WIDTH=256
IMAGE_HEIGHT=256
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    data_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    a=batch_size
)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df,
    data_path,
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# load model without classifier layers
pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
# add new classifier layers
flat1 = Flatten()(pre_model.output)
class1 = Dense(1024, activation='relu')(flat1)
# output = Dense(2, activation='softmax')(class1)
output = Dense(2, activation=None)(class1)
output = Activation('softmax')(output)

# define new model
model = Model(inputs=pre_model.input, outputs=output)

for layer in model.layers[:19]:
    layer.trainable = False

base_learning_rate = 0.0001

model.summary() # show model summary
model.compile(optimizer=tf.keras.optimizers.Adadelta(base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

mdlckpt = ModelCheckpoint(ckpt_path + 'model-{epoch:03d}-{val_acc:03f}.h5', save_best_only=True, monitor='val_acc', mode='max')
callbacks = [earlystop, learning_rate_reduction, mdlckpt]

history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=max(150, total_validate//batch_size),
    steps_per_epoch=max(150,total_train//batch_size),
    callbacks=callbacks
)

model.save_weights(ckpt_path + "{model_name}_weights_keras.h5".format(model_name=model_name))
model.save(ckpt_path + "{model_name}_keras.h5".format(model_name=model_name))

