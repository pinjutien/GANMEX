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

"""Trains a StarGAN model using tfgan.estimator.StarGANEstimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from tensorflow_gan.examples.stargan_estimator import train_lib

# FLAGS for data.
flags.DEFINE_integer('batch_size', 6, 'The number of images in each batch.')
flags.DEFINE_integer('patch_size', 256, 'The patch size of images.')

# Write-to-disk flags.
flags.DEFINE_string('output_dir', '/tmp/tfgan_logdir/stargan_estimator/out/',
                    'Directory where to write summary image.')
flags.DEFINE_string('tfdata_source', 'cycle_gan',
                    'load tf dataset. default=celeb_a')
flags.DEFINE_string('tfdata_source_domains', 'Black_Hair,Blond_Hair,Brown_Hair',
                    'celeb_a domain: default=Black_Hair,Blond_Hair,Brown_Hair')
flags.DEFINE_string('download', "True", "download data from tensorflow_datasets")
flags.DEFINE_string('data_dir', None, "directly load data from data_dir")
flags.DEFINE_string('cls_model', None, "load classification model in discrimnator of stargan")

# FLAGS for training hyper-parameters.
flags.DEFINE_float('generator_lr', 2e-5, 'The generator learning rate. Default = 1e-4')
flags.DEFINE_float('discriminator_lr', 1e-4, 'The discriminator learning rate. Default = 1e-4')
flags.DEFINE_integer('max_number_of_steps', 1000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer('steps_per_eval', 2000,
                     'The number of steps after which we write eval to disk.')
flags.DEFINE_float('adam_beta1', 0.5, 'Adam Beta 1 for the Adam optimizer.')
flags.DEFINE_float('adam_beta2', 0.999, 'Adam Beta 2 for the Adam optimizer.')
flags.DEFINE_float('gen_disc_step_ratio', 1.0,
                   'Generator:Discriminator training step ratio. Default = 0.2')
flags.DEFINE_integer('save_checkpoints_steps', 2000,
                     'Save checkpoint every n step.')
flags.DEFINE_integer('keep_checkpoint_max', 250, 'Max number of checkpoints to keep.')
flags.DEFINE_float('reconstruction_loss_weight', 10.0, 'Default = 10.0')
flags.DEFINE_float('classification_loss_weight', 1.0, 'Default = 1.0')


# FLAGS for distributed training.
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')
flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

FLAGS = flags.FLAGS


def main(_):
  hparams = train_lib.HParams(FLAGS.batch_size, FLAGS.patch_size,
                              FLAGS.output_dir, FLAGS.generator_lr,
                              FLAGS.discriminator_lr, FLAGS.max_number_of_steps,
                              FLAGS.steps_per_eval, FLAGS.adam_beta1,
                              FLAGS.adam_beta2, FLAGS.gen_disc_step_ratio,
                              FLAGS.master, FLAGS.ps_tasks, FLAGS.task,
                              FLAGS.tfdata_source, FLAGS.tfdata_source_domains,
                              FLAGS.download, FLAGS.data_dir, FLAGS.cls_model,
                              FLAGS.save_checkpoints_steps, FLAGS.keep_checkpoint_max,
                              FLAGS.reconstruction_loss_weight, FLAGS.classification_loss_weight)
  train_lib.train(hparams)


if __name__ == '__main__':
  app.run(main)
