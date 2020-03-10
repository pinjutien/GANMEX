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

HParams = collections.namedtuple('HParams', [
  'batch_size', 'patch_size', 'output_dir', 'generator_lr',
  'discriminator_lr', 'max_number_of_steps', 'steps_per_eval', 'adam_beta1',
  'adam_beta2', 'gen_disc_step_ratio', 'master', 'ps_tasks', 'task', 'tfdata_source', 'tfdata_source_domains',
  "model_dir"
])


# def _get_optimizer(gen_lr, dis_lr, beta1, beta2):
#   """Returns generator optimizer and discriminator optimizer.

#   Args:
#     gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
#       rate.
#     dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
#       learning rate.
#     beta1: A scalar float `Tensor` or a Python number. The beta1 parameter to
#       the `AdamOptimizer`.
#     beta2: A scalar float `Tensor` or a Python number. The beta2 parameter to
#       the `AdamOptimizer`.

#   Returns:
#     A tuple of generator optimizer and discriminator optimizer.
#   """
#   gen_opt = tf.compat.v1.train.AdamOptimizer(
#       gen_lr, beta1=beta1, beta2=beta2, use_locking=True)
#   dis_opt = tf.compat.v1.train.AdamOptimizer(
#       dis_lr, beta1=beta1, beta2=beta2, use_locking=True)
#   return gen_opt, dis_opt


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


# def _get_summary_image(estimator, test_images_np):
#     """Returns a numpy image of the generate on the test images."""
#     num_domains = len(test_images_np)

#     img_rows = []
#     for img_np in test_images_np:
      
#         def test_input_fn():
#             dataset_imgs = [img_np] * num_domains  # pylint:disable=cell-var-from-loop
#             dataset_lbls = [tf.one_hot([d], num_domains) for d in xrange(num_domains)]

#             # Make into a dataset.
#             dataset_imgs = np.stack(dataset_imgs)
#             dataset_imgs = np.expand_dims(dataset_imgs, 1)
#             dataset_lbls = tf.stack(dataset_lbls)
#             unused_tensor = tf.zeros(num_domains)
#             return tf.data.Dataset.from_tensor_slices(((dataset_imgs, dataset_lbls),
#                                                        unused_tensor))

#         prediction_iterable = estimator.predict(test_input_fn)
#         predictions = [next(prediction_iterable) for _ in xrange(num_domains)]
#         transform_row = np.concatenate([img_np] + predictions, 1)
#         img_rows.append(transform_row)

#     all_rows = np.concatenate(img_rows, 0)
#     # Normalize` [-1, 1] to [0, 1].
#     normalized_summary = (all_rows + 1.0) / 2.0
#     return normalized_summary

def _get_summary_image(estimator, test_images_np, checkpoint_path):
  """Returns a numpy image of the generate on the test images."""
  num_domains = len(test_images_np)

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
    # res = {}
    # for file_path in tf.io.gfile.glob(input_file_pattern):
    #     filename = file_path.split("/")[-1]
    #     # Grab a single image and run it through inference
    #     input_np = np.asarray(PIL.Image.open(file_path))
    #     res[filename] = input_np
    # return res
        
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

    # with open(hparams.output_dir + 'train_result.json', 'w') as fp:
    #     json.dump(hparams._asdict(), fp, indent=4)

    # # Make sure steps integers are consistent.
    # if hparams.max_number_of_steps % hparams.steps_per_eval != 0:
    #     raise ValueError('`max_number_of_steps` must be divisible by '
    #                     '`steps_per_eval`.')

    # # Create optimizers.
    # gen_opt, dis_opt = _get_optimizer(hparams.generator_lr,
    #                                   hparams.discriminator_lr,
    #                                   hparams.adam_beta1, hparams.adam_beta2)

    # Create estimator.
    # stargan_estimator = tfgan.estimator.StarGANEstimator(
    #     model_dir= hparams.checkpoint_path,
    #     generator_fn=network.generator,
    #     discriminator_fn= network.discriminator,
    #     loss_fn=tfgan.stargan_loss,
    #     generator_optimizer=gen_opt,
    #     discriminator_optimizer=dis_opt,
    #     get_hooks_fn=tfgan.get_sequential_train_hooks(
    #         _define_train_step(hparams.gen_disc_step_ratio)),
    #     add_summaries=tfgan.estimator.SummaryType.IMAGES)


    stargan_estimator = tfgan.estimator.StarGANEstimator(
      model_dir= hparams.model_dir,
      generator_fn=network.generator,
      discriminator_fn=network.discriminator,
      loss_fn=tfgan.stargan_loss,
      add_summaries=tfgan.estimator.SummaryType.IMAGES
    )

    # # Get input function for training and test images.
    # if (hparams.tfdata_source):
    #     print("[**] load tensorflow dataset: {x}".format(x=hparams.tfdata_source))
    #     # train_input_fn = lambda: data_provider.provide_data(  # pylint:disable=g-long-lambda
    #     #     'train', hparams.batch_size, hparams.patch_size, hparams.tfdata_source, hparams.tfdata_source_domains)
    #     test_images_np = data_provider.provide_celeba_test_set(hparams.patch_size)
    # else:
    #     train_input_fn = None
    #     test_images_np = None
    #     raise Exception("TODO: support external data souce.")
      
    # test_images_np_dict = load_image(input_file_pattern)
    # filename_str = os.path.join(hparams.output_dir, 'summary_image_%i.png')
    # for file_path in tf.io.gfile.glob(input_file_pattern):
    #     filename = file_path.split("/")[-1]
    #     test_images_np = load_image(file_path)
    #     summary_img = _get_summary_image(stargan_estimator, test_images_np)
    #     with tf.io.gfile.GFile(filename_str % cur_step, 'w') as f:
    #         PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')
    checkpoint_path = None
    if "checkpoint" in hparams:
        checkpoint_path = hparams.model_dir + hparams.checkpoint
        print("[**] use specific checkpoint: {checkpoint_path}".format(checkpoint_path=checkpoint_path))
    test_images_np = data_provider.provide_celeba_test_set(128)
    summary_img = _get_summary_image(stargan_estimator, test_images_np, checkpoint_path)
    filename_str = os.path.join(hparams.output_dir, 'summary_image.png')
    with tf.io.gfile.GFile(filename_str, 'w') as f:
        PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')
      
    # # Periodically train and write prediction output to disk.
    # cur_step = 0
    # while cur_step < hparams.max_number_of_steps:
    #   cur_step += hparams.steps_per_eval
    #   stargan_estimator.train(train_input_fn, steps=cur_step)
    #   summary_img = _get_summary_image(stargan_estimator, test_images_np)
    #   with tf.io.gfile.GFile(filename_str % cur_step, 'w') as f:
    #     PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')


if __name__ == '__main__':
    input_json = {
      "model_dir": "/Users/ptien/DeepLearning/gan/tensorflow_gan/examples/stargan_estimator/exps/stargan_estimator/checkpoints/",
      # "checkpoint": "model.ckpt-26",
      "output_dir": "./exps/stargan_estimator_output/",
      "patch_size": 128
    }
    if (type(input_json) !=Namespace):
        hparams = Namespace(**input_json)
        
    evaluation(hparams)
