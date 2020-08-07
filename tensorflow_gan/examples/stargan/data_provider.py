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

"""StarGAN data provider."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_gan.examples.cyclegan import data_provider
from tensorflow_gan.examples.cyclegan.data_provider import provide_custom_datasets


def provide_domained_dataset(tfds_name,
                             batch_size,
                             patch_size,
                             split='train',
                             num_parallel_calls=None,
                             shuffle=True,
                             domains=('Black_Hair', 'Blond_Hair', 'Brown_Hair'),
                             download=True,
                             data_dir=None):
  """Provides batches of CelebA image patches (or the datasets with domains).

  Args:
    tfds_name: tensorflow dataset name.
    batch_size: The number of images in each batch.
    patch_size: Python int. The patch size to extract.
    split: Either 'train' or 'test'.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    domains: Name of domains to transform between. Must be in Celeb A dataset.

  Returns:
    A tf.data.Dataset with:
      * images:  `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
          Output pixel values are in [-1, 1].
      * labels: A `Tensor` of size [batch_size, 10] of one-hot label
          encodings with type tf.int32, or a `Tensor` of size [batch_size],
          depending on the value of `one_hot`.

  Raises:
    ValueError: If `split` isn't `train` or `test`.
  """
  # ds = tfds.load('celeb_a', split=split, shuffle_files=shuffle)
  print("[**] Load tf data source: {tfdata_source}".format(tfdata_source=tfds_name))
  ds = tfds.load(tfds_name, split=split, shuffle_files=shuffle, download=download, data_dir=data_dir)

  def _filter_pred(attribute):
    def _filter(element):
      return element['attributes'][attribute]
    return _filter
  dss = tuple([ds.filter(_filter_pred(attribute)) for attribute in domains])
  ds = tf.data.Dataset.zip(dss)

  def _preprocess(*elements):
    """Map elements to the example dicts expected by the model."""
    output_dict = {}
    num_domains = len(elements)
    for idx, (domain, elem) in enumerate(zip(domains, elements)):
      uint8_img = elem['image']
      patch = data_provider.full_image_to_patch(uint8_img, patch_size)
      label = tf.one_hot(idx, num_domains)
      output_dict[domain] = {'images': patch, 'labels': label}
    return output_dict

  ds = (ds
        .map(_preprocess, num_parallel_calls=num_parallel_calls)
        .cache()
        .repeat())
  if shuffle:
    ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
  ds = (ds
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

  return ds


def provide_categorized_dataset(tfds_name,
                                batch_size,
                                patch_size,
                                split='train',
                                num_parallel_calls=None,
                                shuffle=True,
                                download=True,
                                data_dir=None):
    """Provides batches of tensorflow dataset the comes with classification labels.

    Args:
      tfds_name: tensorflow dataset name.
      batch_size: The number of images in each batch.
      patch_size: Python int. The patch size to extract.
      num_classes: number for classes.
      split: Either 'train' or 'test'.
      num_parallel_calls: Number of threads dedicated to parsing.
      shuffle: Whether to shuffle.

    Returns:
      A tf.data.Dataset with:
        * images:  `Tensor` of size [batch_size, 32, 32, 3] and type tf.float32.
            Output pixel values are in [-1, 1].
        * labels: A `Tensor` of size [batch_size, 10] of one-hot label
            encodings with type tf.int32, or a `Tensor` of size [batch_size],
            depending on the value of `one_hot`.

    Raises:
      ValueError: If `split` isn't `train` or `test`.
    """

    print("[**] Load tf data source: {tfdata_source}".format(tfdata_source=tfds_name))
    ds, info = tfds.load(tfds_name, split=split, shuffle_files=shuffle, with_info=True, download=download, data_dir=data_dir)

    if set(info.features.keys()) != set(['image', 'label']):
        raise NotImplementedError('Unsupported tfds name:' + tfds_name)

    num_classes = info.features['label'].num_classes
    num_channels = info.features['image'].shape[2]
    image_size_max = max(info.features['image'].shape[:2])
    if patch_size > image_size_max:
        print('Raw image shape is %s. Capping the patch_size at %d' % (str(info.features['image'].shape), image_size_max))
        patch_size = image_size_max

    def _filter_pred(label):
        def _filter(element):
            # if element['label'] not in list(range(10)):
            #     raise Exception('Cant recognize label!!!' + str(element['label']))
            return tf.math.equal(element['label'], label)
        return _filter

    dss = tuple([ds.filter(_filter_pred(label)) for label in range(num_classes)])
    ds = tf.data.Dataset.zip(dss)

    def _preprocess(*elements):
        """Map elements to the example dicts expected by the model."""
        output_dict = {}
        for idx, elem in enumerate(elements):
            uint8_img = elem['image']
            patch = data_provider.full_image_to_patch(uint8_img, patch_size, num_channels)
            label = tf.one_hot(idx, num_classes)
            output_dict[idx] = {'images': patch, 'labels': label}
        return output_dict

    ds = (ds
          .map(_preprocess, num_parallel_calls=num_parallel_calls)
          .cache()
          .repeat())
    if shuffle:
        ds = ds.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
    ds = (ds
          .batch(batch_size, drop_remainder=True)
          .prefetch(tf.data.experimental.AUTOTUNE))

    return ds


def provide_data(tfds_name,
                 batch_size,
                 patch_size,
                 split='train',
                 num_parallel_calls=None,
                 shuffle=True,
                 domains=('Black_Hair', 'Blond_Hair', 'Brown_Hair'),
                 download=True,
                 data_dir=None):
    """Provides batches of CelebA image patches.

    Args:
    tfds_name: tensorflow dataset name.
    batch_size: The number of images in each batch.
    patch_size: Python int. The patch size to extract.
    split: Either 'train' or 'test'.
    num_parallel_calls: Number of threads dedicated to parsing.
    shuffle: Whether to shuffle.
    domains: Name of domains to transform between. Must be in Celeb A dataset.

    Returns:
    A tf.data.Dataset with:
      * images:  `Tensor` of size [batch_size, patch_size, patch_size, 3] and
          type tf.float32. Output pixel values are in [-1, 1].
      * labels: A `Tensor` of size [batch_size, 10] of one-hot label
          encodings with type tf.int32, or a `Tensor` of size [batch_size],
          depending on the value of `one_hot`.

    Raises:
    ValueError: If `split` isn't `train` or `test`.
    """

    if tfds_name.startswith('cycle_gan'):
        ds = provide_custom_datasets(batch_size,
                                     None,
                                     shuffle,
                                     1,
                                     patch_size,
                                     tfds_name,
                                     with_labels=True)

        images = [d['images'] for d in ds]
        labels = [d['labels'] for d in ds]

    else:
        if tfds_name.startswith('celeb_a'):
            ds = provide_domained_dataset(tfds_name, batch_size, patch_size,
                                          split=split,
                                          num_parallel_calls=num_parallel_calls,
                                          shuffle=shuffle,
                                          domains=domains,
                                          download=download,
                                          data_dir=data_dir)

        else:
            ds = provide_categorized_dataset(tfds_name, batch_size, patch_size,
                                             split=split,
                                             num_parallel_calls=num_parallel_calls,
                                             shuffle=shuffle,
                                             download=download,
                                             data_dir=data_dir)

        next_batch = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
        domains = next_batch.keys()
        images = [next_batch[domain]['images'] for domain in domains]
        labels = [next_batch[domain]['labels'] for domain in domains]

    return images, labels

