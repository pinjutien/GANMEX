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

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_gan.examples.cyclegan import data_provider as cyclegan_dp
from tensorflow_gan.examples.stargan import data_provider
import PIL

provide_data = data_provider.provide_data


def provide_celeba_test_set(patch_size, download, data_dir, num_images=3):
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


def provide_categorized_test_set(tfds_name, patch_size, download, data_dir, num_images=6):
    """Provide one example of every class.

    Args:
      patch_size: Python int. The patch size to extract.

    Returns:
      An `np.array` of shape (num_domains, H, W, C) representing the images.
        Values are in [-1, 1].
    """
    ds, info = tfds.load(tfds_name, download=download, data_dir=data_dir, split='test', with_info=True)
    num_classes = info.features['label'].num_classes
    num_channels = info.features['image'].shape[2]
    image_size_max = max(info.features['image'].shape[:2])
    if patch_size > image_size_max:
        print('Raw image shape is %s. Capping the patch_size at %d' % (str(info.features['image'].shape), image_size_max))
        patch_size = image_size_max

    def _preprocess(x):
      return {
          'image': cyclegan_dp.full_image_to_patch(x['image'], patch_size, num_channels),
          'label': x['label'],
      }
    ds = ds.map(_preprocess)
    ds_np = tfds.as_numpy(ds)

    # Get one image of each hair type.
    images = []
    labels = []
    targets = set()
    while len(images) < num_images:
        if not targets:
            targets = set(range(num_classes))

        elem = next(ds_np)
        if elem['label'] in targets:
            targets.remove(elem['label'])

            images.append(elem['image'])
            labels.append(tf.one_hot(elem['label'], num_classes))

    images = np.array(images, dtype=np.float32)

    assert images.dtype == np.float32
    assert np.max(np.abs(images)) <= 1.0
    assert images.shape == (num_images, patch_size, patch_size, num_channels)

    return images, num_classes


def provide_cyclegan_test_set(patch_size, num_images=6):
  """Provide one example of every class.

  Args:
    patch_size: Python int. The patch size to extract.

  Returns:
    An `np.array` of shape (num_domains, H, W, C) representing the images.
      Values are in [-1, 1].
  """
  ds = tfds.load('cycle_gan')

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

# def provide_celeba_test_set1(patch_size):
#   """Provide one example of every class.

#   Args:
#     patch_size: Python int. The patch size to extract.

#   Returns:
#     An `np.array` of shape (num_domains, H, W, C) representing the images.
#       Values are in [-1, 1].
#   """
#   ds = tfds.load('celeb_a', split='test')
#   def _preprocess(x):
#     return {
#         'image': cyclegan_dp.full_image_to_patch(x['image'], patch_size), # x['image']
#         'attributes': x['attributes'],
#     }
#   ds = ds.map(_preprocess)
#   ds_np = tfds.as_numpy(ds)

#   # Get one image of each hair type.
#   images = []
#   labels = []
#   while len(images) < 3:
#     elem = next(ds_np)
#     attr = elem['attributes']
#     cur_lbl = [attr['Black_Hair'], attr['Blond_Hair'], attr['Brown_Hair']]
#     if cur_lbl not in labels:
#       # output_path = "black_hair_{a1}_blond_hair_{a2}_brown_hair_{a3}.png".format(a1=int(attr['Black_Hair']),
#       #                                                                            a2=int(attr['Blond_Hair']),
#       #                                                                            a3=int(attr['Brown_Hair']))
#       # PIL.Image.fromarray(elem['image']).save("./testdata/" + output_path)
#       images.append(elem['image'])
#       labels.append(cur_lbl)
#   images = np.array(images, dtype=np.float32)
#   labels = np.array(labels)
  
#   assert images.dtype == np.float32
#   assert np.max(np.abs(images)) <= 1.0
#   assert images.shape == (3, patch_size, patch_size, 3)
#   return images, labels
