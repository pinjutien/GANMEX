# coding=utf-8
# Copyright 2020 The TensorFlow GAN Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""StarGAN Estimator data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import numpy as np
from scipy.io import loadmat

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_gan.examples.cyclegan import data_provider as cyclegan_dp
from tensorflow_gan.examples.stargan import data_provider
from tensorflow_gan import custom_tfds
import PIL

provide_data = data_provider.provide_data


def provide_celeba_test_set(patch_size, download=True, data_dir=None, num_images=3):
  """Provide one example of every class.

  Args:
    patch_size: Python int. The patch size to extract.

  Returns:
    An `np.array` of shape (num_domains, H, W, C) representing the images.
      Values are in [-1, 1].
  """
  ds = tfds.load('celeb_a', download=download, data_dir=data_dir, split='test')
  def _preprocess(x):
    return {
        'image': cyclegan_dp.full_image_to_patch(x['image'], patch_size),
        'attributes': x['attributes'],
    }
  ds = ds.map(_preprocess)
  ds_np = tfds.as_numpy(ds)

  # Get one image of each hair type.
  images = []
  labels = []
  while len(images) < num_images:
    elem = next(ds_np)
    attr = elem['attributes']
    cur_lbl = [attr['Black_Hair'], attr['Blond_Hair'], attr['Brown_Hair']]
    if cur_lbl not in labels:
      images.append(elem['image'])
      labels.append(cur_lbl)
  images = np.array(images, dtype=np.float32)

  assert images.dtype == np.float32
  assert np.max(np.abs(images)) <= 1.0
  assert images.shape == (num_images, patch_size, patch_size, 3)

  return images


def provide_categorized_test_set(tfds_name, patch_size, color_labeled=0, download=True, data_dir=None,
                                 num_images=20, filtered_label=None):
    """Provide one example of every class.

    Args:
      patch_size: Python int. The patch size to extract.
      filtered_label: None for all

    Returns:
      An `np.array` of shape (num_domains, H, W, C) representing the images.
        Values are in [-1, 1].
    """
    split = 'train' if tfds_name in ('plant_village_v0', 'plant_village_v1', 'kaggle_birds_v0', 'kaggle_birds_v1') \
                else 'test'
    ds, info = tfds.load(tfds_name, download=download, data_dir=data_dir, split=split, with_info=True)
    num_classes = info.features['label'].num_classes
    num_channels = info.features['image'].shape[2]
    if info.features['image'].shape[0] is not None and info.features['image'].shape[1] is not None:
        image_size_max = max(info.features['image'].shape[:2])
        if patch_size > image_size_max:
            print('Raw image shape is %s. Capping the patch_size at %d' % (str(info.features['image'].shape), image_size_max))
            patch_size = image_size_max

    if color_labeled and num_channels == 1:
        if color_labeled == 2:
            num_classes = 3
        def _color_label(element):
            bw_image = element['image']
            blank_image = tf.zeros(bw_image.get_shape(), dtype=tf.uint8)
            image_sum = tf.cast(tf.reduce_sum(bw_image), tf.int32)
            color = tf.math.floormod(image_sum, 3)
            image = [(bw_image if tf.math.equal(color, x) else blank_image) for x in range(3)]
            image = tf.concat(image, axis=2)

            if color_labeled == 2:
                label = color
            else:
                label = element['label']

            return {'image': image, 'label': label}
        ds = ds.map(_color_label)
        num_channels = 3

    def _preprocess(x):
      return {
          'image': cyclegan_dp.full_image_to_patch(x['image'], patch_size, num_channels),
          'label': x['label'],
      }
    ds = ds.map(_preprocess)
    ds_np = tfds.as_numpy(ds)

    # Get one image of each hair type.
    images = []

    if filtered_label is None:
        targets = set()
        while len(images) < num_images:
            if not targets:
                targets = set(range(num_classes))

            elem = next(ds_np)
            if elem['label'] in targets:
                targets.remove(elem['label'])

                images.append(elem['image'])
    else:
        while len(images) < num_images:
            elem = next(ds_np)
            if elem['label'] == filtered_label:
                images.append(elem['image'])

    images = np.array(images, dtype=np.float32)

    assert images.dtype == np.float32
    assert np.max(np.abs(images)) <= 1.0
    assert images.shape == (num_images, patch_size, patch_size, num_channels)

    return images, num_classes


def provide_cyclegan_test_set(tfds_name, patch_size, num_images=6):
  """Provide one example of every class.

  Args:
    tfds_name: string, tfds name
    patch_size: Python int. The patch size to extract.

  Returns:
    An `np.array` of shape (num_domains, H, W, C) representing the images.
      Values are in [-1, 1].
  """
  ds = tfds.load(tfds_name)

  num_images_B = num_images // 2
  num_images_A = num_images - num_images_B

  examples_A = list(tfds.as_numpy(ds['testA'].take(num_images_A)))
  examples_B = list(tfds.as_numpy(ds['testB'].take(num_images_B)))

  images = [tfds.as_numpy(cyclegan_dp.full_image_to_patch(x['image'], patch_size)) for x in examples_A + examples_B]
  images = np.array(images, dtype=np.float32)

  assert images.dtype == np.float32
  assert np.max(np.abs(images)) <= 1.0
  assert images.shape == (num_images, patch_size, patch_size, 3)

  return images
