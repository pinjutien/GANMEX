import getpass
import os
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from PIL import Image
from keras.datasets import cifar10
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

import tensorflow as tf
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


def get_datasets(dataset_name, target_shape, test_run=False, train_val_split=True, fill_color=False):

    ds, info = tfds.load(dataset_name, with_info=True)
    num_classes = info.features['label'].num_classes

    raw_struct = {}
    if dataset_name.startswith('cycle_gan'):
        raw_struct['train'] = list(tfds.as_numpy(ds['trainA'])) + list(tfds.as_numpy(ds['trainB']))
        raw_struct['test'] = list(tfds.as_numpy(ds['testA'])) + list(tfds.as_numpy(ds['testB']))
    else:
        if fill_color:
            def _color_label(element):
                bw_image = element['image']
                blank_image = tf.zeros(bw_image.get_shape(), dtype=tf.uint8)
                image_sum = tf.cast(tf.reduce_sum(bw_image), tf.int32)
                color = tf.math.floormod(image_sum, 3)
                image = [(bw_image if tf.math.equal(color, x) else blank_image) for x in range(3)]
                image = tf.concat(image, axis=2)
                return {'image': image, 'label': element['label']}

            ds['train'] = ds['train'].map(_color_label)
            ds['test'] = ds['test'].map(_color_label)

        raw_struct['train'] = list(tfds.as_numpy(ds['train']))
        raw_struct['test'] = list(tfds.as_numpy(ds['test']))

    if train_val_split:
        raw_struct['train'], raw_struct['val'] = train_test_split(raw_struct['train'], test_size=0.20, random_state=42)

    data_struct = {}
    input_shape = (info.features['image'].shape[0], info.features['image'].shape[1])
    for key, raw_data in raw_struct.items():
        if test_run:
            raw_data = raw_data[100]

        if input_shape != target_shape:
            curr_x = np.array([np.array(Image.fromarray(a['image']).resize(target_shape)) for a in raw_data])
        else:
            curr_x = np.array([a['image'] for a in raw_data])

        curr_y = np.array([a['label'] for a in raw_data])
        curr_y = np_utils.to_categorical(curr_y, num_classes)

        data_struct[key] = (curr_x, curr_y)

    return data_struct


def get_generators(dataset_name, data_struct, batch_size):
    # data_struct = get_datasets(dataset_name, target_shape, test_run=test_run, train_val_split=train_val_split)

    generator_struct = {}
    for key, (curr_x, curr_y) in data_struct.items():
        if key == 'train':
            if dataset_name == 'mnist':
                datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=5,
                    shear_range=0.1,
                    zoom_range=0.1,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                )
            elif dataset_name == 'svhn_cropped':
                datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    shear_range=0.3,
                    zoom_range=0.1,
                    height_shift_range=0.1,
                )
            elif dataset_name == 'cycle_gan':
                datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=45,
                    shear_range=0.1,
                    zoom_range=0.3,
                    horizontal_flip=True,
                    vertical_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                )
            elif dataset_name == 'cifar10':
                datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=15,
                    shear_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=False,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                )
            else:
                datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=15,
                    shear_range=0.1,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    vertical_flip=True,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                )
        else:
            datagen = ImageDataGenerator(rescale=1. / 255)

        curr_batch_size = batch_size if key != 'test' else 1
        generator_struct[key] = {
            'generator': datagen.flow(curr_x, curr_y, batch_size=curr_batch_size),
            'steps': len(curr_x) // curr_batch_size
        }

    return generator_struct


def get_generators_from_tfds(dataset_name, target_shape, batch_size, test_run=False, train_val_split=True):
    ds, info = tfds.load(dataset_name, with_info=True)
    num_classes = info.features['label'].num_classes

    if dataset_name.startswith('cycle_gan'):
        raw_train = list(tfds.as_numpy(ds['trainA'])) + list(tfds.as_numpy(ds['trainB']))
    else:
        raw_train = list(tfds.as_numpy(ds['train']))

    if train_val_split:
        raw_train, raw_val = train_test_split(raw_train, test_size=0.20, random_state=42)

    if test_run:
        raw_train = raw_train[:batch_size]

    train_size = len(raw_train)

    input_shape = (info.features['image'].shape[0], info.features['image'].shape[1])
    if input_shape != target_shape:
        x_train = np.array([np.array(Image.fromarray(a['image']).resize(target_shape)) for a in raw_train])
    else:
        x_train = np.array([a['image'] for a in raw_train])

    y_train = np.array([a['label'] for a in raw_train])
    y_train = np_utils.to_categorical(y_train, num_classes)

    if dataset_name == 'mnist':
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
        )
    else:
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