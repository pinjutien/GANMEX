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

from tensorflow_gan.examples.stargan import network
from tensorflow_gan.examples.stargan_estimator import train_lib

# FLAGS for data.
flags.DEFINE_integer('batch_size', 6, 'The number of images in each batch.')
flags.DEFINE_integer('patch_size', 256, 'The patch size of images.')

## Next cw100 scl100
## tweak gen_disc_step_ratio


# Write-to-disk flags.
flags.DEFINE_string('output_dir',
                    '/tmp/tfgan_logdir_keras_rmsp_v1_cw1000_sw100_rw0/stargan_estimator/out/',
                    # '/tmp/tfgan_rps_test_p128_gd02/stargan_estimator/out/',
                    # '/tmp/tfgan_logdir_keras_rmsp_v1_cw100_scl100_smooth/stargan_estimator/out/',
                    # '/tmp/stargan_logdir_gentest_p128_scl0_smoothBL_cks3/stargan_estimator/out/',
                    # '/tmp/tfgan_logdir_p128_hack1/stargan_estimator/out/',
                    'Directory where to write summary image.')
flags.DEFINE_string('tfdata_source', 'cycle_gan',
                    'load tf dataset. celeb_a, cycle_gan, mnist, rock_paper_scissors')
flags.DEFINE_string('tfdata_source_domains', 'Black_Hair,Blond_Hair,Brown_Hair',
                    'celeb_a domain: default=Black_Hair,Blond_Hair,Brown_Hair')
flags.DEFINE_string('download', "True", "download data from tensorflow_datasets")
flags.DEFINE_string('data_dir', None, "directly load data from data_dir")
flags.DEFINE_string('cls_model',
                    # None,
                    "/home/ec2-user/gan/test_model/rmsp_std_conv1_gmp_ds1024_dbn/",
                    # '/Users/shengms/Code/gan/tensorflow_gan/examples/classification/test_model/test_a2o/',
                    "load classification model in discriminator of stargan")
flags.DEFINE_string('cls_checkpoint',
                    None,
                    # '/home/ec2-user/gan_checkpoints/tfgan_logdir_glr2m5_gd1_ab09/stargan_estimator/out/checkpoints/model.ckpt-130000',
                    "checkpoint file path for the class discriminator")

# FLAGS for training hyper-parameters.
flags.DEFINE_float('generator_lr', 2e-5, 'The generator learning rate. Default = 1e-4 Current Best = 2e-5')
flags.DEFINE_float('discriminator_lr', 1e-4, 'The discriminator learning rate. Default = 1e-4')
flags.DEFINE_integer('max_number_of_steps', 1000000,
                     'The maximum number of gradient steps.')
flags.DEFINE_integer('steps_per_eval', 2000,
                     'The number of steps after which we write eval to disk.')
flags.DEFINE_float('adam_beta1', 0.9, 'Adam Beta 1 for the Adam optimizer. Default = 0.5 Current Best = 0.9')
flags.DEFINE_float('adam_beta2', 0.999, 'Adam Beta 2 for the Adam optimizer. Default = 0.999')
flags.DEFINE_float('gen_disc_step_ratio', 1.0,
                   'Generator:Discriminator training step ratio. Default = 0.2 Current Best = 1.0')
flags.DEFINE_integer('save_checkpoints_steps', 2000,
                     'Save checkpoint every n step.')
flags.DEFINE_integer('keep_checkpoint_max', 250, 'Max number of checkpoints to keep.')

flags.DEFINE_float('reconstruction_loss_weight', 0.0, 'Default = 10.0')
flags.DEFINE_float('self_consistency_loss_weight', 100.0, 'Put in 0.0 if not in use. Current Best = 1000.0')
flags.DEFINE_float('classification_loss_weight', 1000.0, 'Default = 1.0')

flags.DEFINE_integer('use_color_labels', 1,
                     'Fill in RGB colors for black and white dataset. 1: original labels, 2: color labels')

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
                              FLAGS.download, FLAGS.data_dir, FLAGS.cls_model, FLAGS.cls_checkpoint,
                              FLAGS.save_checkpoints_steps, FLAGS.keep_checkpoint_max,
                              FLAGS.reconstruction_loss_weight,
                              FLAGS.self_consistency_loss_weight,
                              FLAGS.classification_loss_weight,
                              FLAGS.use_color_labels)

  override_generator_fn = None
  # override_generator_fn = network.generator_hack
  # override_generator_fn = network.generator_smooth

  train_lib.train(hparams, override_generator_fn=override_generator_fn)


if __name__ == '__main__':
  app.run(main)
