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

"""evaluation a StarGAN model using tfgan.estimator.StarGANEstimator. train_lib.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import json
import numpy as np
import PIL
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_gan as tfgan
from argparse import Namespace
from tensorflow_gan.examples.stargan import network
from tensorflow_gan.examples.stargan_estimator import data_provider
from data_helper import load_custom_data

HParams = collections.namedtuple('HParams', [
  'batch_size', 'patch_size', 'output_dir', 'generator_lr',
  'discriminator_lr', 'max_number_of_steps', 'steps_per_eval', 'adam_beta1',
  'adam_beta2', 'gen_disc_step_ratio', 'master', 'ps_tasks', 'task', 'tfdata_source', 'tfdata_source_domains',
  "model_dir"
])

def _define_train_step(gen_disc_step_ratio):
    """Get the training step for generator and discriminator for each GAN step.

    Args:
      gen_disc_step_ratio: A python number. The ratio of generator to
        discriminator training steps.

    Returns:
      GANTrainSteps namedtuple representing the training step configuration.
    """

    if gen_disc_step_ratio <= 1:
        discriminator_step = int(1 / gen_disc_step_ratio)
        return tfgan.GANTrainSteps(1, discriminator_step)
    else:
        generator_step = int(gen_disc_step_ratio)
        return tfgan.GANTrainSteps(generator_step, 1)


def _get_summary_image(estimator, test_images_np, checkpoint_path, num_domains):
    """Returns a numpy image of the generate on the test images."""
    # num_domains = len(test_images_np)

    img_rows = []
    for img_np in test_images_np:
        def test_input_fn():
            dataset_imgs = [img_np] * num_domains  # pylint:disable=cell-var-from-loop
            dataset_lbls = [tf.one_hot([d], num_domains) for d in xrange(num_domains)]

            # Make into a dataset.
            dataset_imgs = np.stack(dataset_imgs)
            dataset_imgs = np.expand_dims(dataset_imgs, 1)
            dataset_lbls = tf.stack(dataset_lbls)
            unused_tensor = tf.zeros(num_domains)
            return tf.data.Dataset.from_tensor_slices(((dataset_imgs, dataset_lbls),
                                                       unused_tensor))

        prediction_iterable = estimator.predict(test_input_fn, checkpoint_path=checkpoint_path)
        predictions = [next(prediction_iterable) for _ in xrange(num_domains)]
        transform_row = np.concatenate([img_np] + predictions, 1)
        img_rows.append(transform_row)

    all_rows = np.concatenate(img_rows, 0)
    # Normalize` [-1, 1] to [0, 1].
    normalized_summary = (all_rows + 1.0) / 2.0
    return normalized_summary

def load_image(file_path):
    filename = file_path.split("/")[-1]
    # Grab a single image and run it through inference
    input_np = np.asarray(PIL.Image.open(file_path))
    return input_np
        
def evaluation(hparams):
    """evaluation a StarGAN.

    Args:
      hparams: An HParams instance containing the hyperparameters for training.
      override_generator_fn: A generator function that overrides the default one.
      override_discriminator_fn: A discriminator function that overrides the
        default one.
    """
    # Create directories if not exist.
    if not tf.io.gfile.exists(hparams.output_dir):
        tf.io.gfile.makedirs(hparams.output_dir)

    stargan_estimator = tfgan.estimator.StarGANEstimator(
      model_dir= hparams.model_dir,
      generator_fn=network.generator,
      discriminator_fn=network.discriminator,
      loss_fn=tfgan.stargan_loss,
      add_summaries=tfgan.estimator.SummaryType.IMAGES
    )

    if (hparams.input_data is None):
        print("[**] load tensorflow dataset as test images.")
        test_images_np = data_provider.provide_celeba_test_set(hparams.patch_size)
    else:
        print("[**] load test images from {input_data}".format(input_data=hparams.input_data))
        test_images_np, test_images_name_list, test_labels_list = load_custom_data(hparams.input_data, hparams.patch_size)

    checkpoint_path = None
    if "checkpoint" in hparams:
        checkpoint_path = hparams.model_dir + hparams.checkpoint
        print("[**] use specific checkpoint: {checkpoint_path}".format(checkpoint_path=checkpoint_path))
    num_domains=len(tuple(hparams.tfdata_source_domains.split(",")))
    summary_img = _get_summary_image(stargan_estimator, test_images_np, checkpoint_path, num_domains)
    filename_str = os.path.join(hparams.output_dir, 'summary_image.png')
    with tf.io.gfile.GFile(filename_str, 'w') as f:
        PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')
      

if __name__ == '__main__':
    input_json = {
      "model_dir": "/Users/ptien/DeepLearning/gan/tensorflow_gan/examples/stargan_estimator/exps/stargan_estimator/checkpoints/",
      # "checkpoint": "model.ckpt-26",
      "input_data": "./testdata/*.png",
      "output_dir": "./exps/stargan_estimator_output5/",
      "tfdata_source_domains": 'Black_Hair,Blond_Hair,Brown_Hair',
      "patch_size": 128
    }
    # with open('./exps/stargan_estimator/train_result.json') as f:
    #   data = json.load(f)
    
    if (type(input_json) !=Namespace):
        hparams = Namespace(**input_json)
        
    evaluation(hparams)
