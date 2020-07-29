import os
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import tensorflow.keras.utils as np_utils


def get_tfds_df_path(data_path):
    filenames = []
    categories = []
    label_map = {
        'trainA': 'horse',
        'trainB': 'zebra',
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


def get_generators_from_df_path(df, data_path, target_size, batch_size):
    train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_size = train_df.shape[0]
    val_size = val_df.shape[0]

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        data_path,
        x_col='filename',
        y_col='category',
        target_size=target_size,
        class_mode='categorical',
        a=batch_size
    )

    val_generator = validation_datagen.flow_from_dataframe(
        val_df,
        data_path,
        x_col='filename',
        y_col='category',
        target_size=target_size,
        class_mode='categorical',
        batch_size=batch_size
    )

    return train_generator, val_generator, train_size, val_size


def get_generators_from_tfds(dataset_name, batch_size, num_classes=2, test_run=False, train_val_split=True):
    ds = tfds.load(dataset_name)
    raw_train = list(tfds.as_numpy(ds['trainA'])) + list(tfds.as_numpy(ds['trainB']))

    if train_val_split:
        raw_train, raw_val = train_test_split(raw_train, test_size=0.20, random_state=42)

    if test_run:
        raw_train = raw_train[:batch_size]

    train_size = len(raw_train)

    x_train = np.array([a['image'] for a in raw_train])
    y_train = np.array([a['label'] for a in raw_train])
    y_train = np_utils.to_categorical(y_train, num_classes)

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    train_generator = train_datagen.flow(x_train, y_train, batch_size=32)

    if train_val_split:
        if test_run:
            raw_val = raw_val[:batch_size]

        val_size = len(raw_val)

        x_val = np.array([a['image'] for a in raw_val])
        y_val = np.array([a['label'] for a in raw_val])
        y_val = np_utils.to_categorical(y_val, num_classes)

        val_datagen = ImageDataGenerator(rescale=1. / 255)
        val_generator = val_datagen.flow(x_val, y_val, batch_size=32)

        return train_generator, val_generator, train_size, val_size
    else:
        return train_generator, train_size


def get_generators_from_cifar10(batch_size):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)