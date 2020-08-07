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

import collections
import os
import json
import numpy as np
import PIL
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_gan as tfgan

from tensorflow_gan.examples.stargan import network
from tensorflow_gan.examples.stargan_estimator import data_provider

HParams = collections.namedtuple('HParams', [
    'batch_size', 'patch_size', 'output_dir', 'generator_lr',
    'discriminator_lr', 'max_number_of_steps', 'steps_per_eval', 'adam_beta1',
    'adam_beta2', 'gen_disc_step_ratio', 'master', 'ps_tasks', 'task', 'tfdata_source', 'tfdata_source_domains',
    'download', 'data_dir', 'cls_model', 'cls_checkpoint', 'save_checkpoints_steps', 'keep_checkpoint_max',
    'reconstruction_loss_weight', 'self_consistency_loss_weight', 'classification_loss_weight'])


def _get_optimizer(gen_lr, dis_lr, beta1, beta2):
  """Returns generator optimizer and discriminator optimizer.

  Args:
    gen_lr: A scalar float `Tensor` or a Python number.  The Generator learning
      rate.
    dis_lr: A scalar float `Tensor` or a Python number.  The Discriminator
      learning rate.
    beta1: A scalar float `Tensor` or a Python number. The beta1 parameter to
      the `AdamOptimizer`.
    beta2: A scalar float `Tensor` or a Python number. The beta2 parameter to
      the `AdamOptimizer`.

  Returns:
    A tuple of generator optimizer and discriminator optimizer.
  """
  gen_opt = tf.compat.v1.train.AdamOptimizer(
      gen_lr, beta1=beta1, beta2=beta2, use_locking=True)
  dis_opt = tf.compat.v1.train.AdamOptimizer(
      dis_lr, beta1=beta1, beta2=beta2, use_locking=True)
  return gen_opt, dis_opt


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


def _get_summary_image(estimator, test_images_np, num_domains):
  """Returns a numpy image of the generate on the test images."""

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

    prediction_iterable = estimator.predict(test_input_fn)
    predictions = [next(prediction_iterable) for _ in xrange(num_domains)]
    transform_row = np.concatenate([img_np] + predictions, 1)
    img_rows.append(transform_row)

  all_rows = np.concatenate(img_rows, 0)
  # Normalize` [-1, 1] to [0, 1].
  normalized_summary = (all_rows + 1.0) / 2.0
  return normalized_summary


from tensorflow_gan.python.train import tuple_losses, losses_wargs, _use_aux_loss, namedtuples

def _get_stargan_loss(
    generator_loss_fn=tuple_losses.stargan_generator_loss_wrapper(
        losses_wargs.wasserstein_generator_loss),
    discriminator_loss_fn=tuple_losses.stargan_discriminator_loss_wrapper(
        losses_wargs.wasserstein_discriminator_loss),
    gradient_penalty_weight=10.0,
    gradient_penalty_epsilon=1e-10,
    gradient_penalty_target=1.0,
    gradient_penalty_one_sided=False,
    reconstruction_loss_fn=tf.compat.v1.losses.absolute_difference,
    reconstruction_loss_weight=10.0,
    self_consistency_loss_fn=tf.compat.v1.losses.absolute_difference,
    self_consistency_loss_weight=0.0,
    classification_loss_fn=tf.compat.v1.losses.softmax_cross_entropy,
    classification_loss_weight=1.0,
    classification_one_hot=True,
    add_summaries=True):

    def stargan_loss(model):
      """StarGAN Loss.

      Args:
        model: (StarGAN) Model output of the stargan_model() function call.
        generator_loss_fn: The loss function on the generator. Takes a
          `StarGANModel` named tuple.
        discriminator_loss_fn: The loss function on the discriminator. Takes a
          `StarGANModel` namedtuple.
        gradient_penalty_weight: (float) Gradient penalty weight. Default to 10 per
          the original paper https://arxiv.org/abs/1711.09020. Set to 0 or None to
          turn off gradient penalty.
        gradient_penalty_epsilon: (float) A small positive number added for
          numerical stability when computing the gradient norm.
        gradient_penalty_target: (float, or tf.float `Tensor`) The target value of
          gradient norm. Defaults to 1.0.
        gradient_penalty_one_sided: (bool) If `True`, penalty proposed in
          https://arxiv.org/abs/1709.08894 is used. Defaults to `False`.
        reconstruction_loss_fn: The reconstruction loss function. Default to L1-norm
          and the function must conform to the `tf.losses` API.
        reconstruction_loss_weight: Reconstruction loss weight. Default to 10.0.
        classification_loss_fn: The loss function on the discriminator's ability to
          classify domain of the input. Default to one-hot softmax cross entropy
          loss, and the function must conform to the `tf.losses` API.
        classification_loss_weight: (float) Classification loss weight. Default to
          1.0.
        classification_one_hot: (bool) If the label is one hot representation.
          Default to True. If False, classification classification_loss_fn need to
          be sigmoid cross entropy loss instead.
        add_summaries: (bool) Add the loss to the summary

      Returns:
        GANLoss namedtuple where we have generator loss and discriminator loss.

      Raises:
        ValueError: If input StarGANModel.input_data_domain_label does not have rank
        2, or dimension 2 is not defined.
      """

      def _classification_loss_helper(true_labels, predict_logits, scope_name):
        """Classification Loss Function Helper.

        Args:
          true_labels: Tensor of shape [batch_size, num_domains] representing the
            label where each row is an one-hot vector.
          predict_logits: Tensor of shape [batch_size, num_domains] representing the
            predicted label logit, which is UNSCALED output from the NN.
          scope_name: (string) Name scope of the loss component.

        Returns:
          Single scalar tensor representing the classification loss.
        """

        with tf.compat.v1.name_scope(
            scope_name, values=(true_labels, predict_logits)):

          loss = classification_loss_fn(
              onehot_labels=true_labels, logits=predict_logits)

          if not classification_one_hot:
            loss = tf.reduce_sum(input_tensor=loss, axis=1)
          loss = tf.reduce_mean(input_tensor=loss)

          if add_summaries:
            tf.compat.v1.summary.scalar(scope_name, loss)

          return loss

      # Check input shape.
      model.input_data_domain_label.shape.assert_has_rank(2)
      model.input_data_domain_label.shape[1:].assert_is_fully_defined()

      # Adversarial Loss.
      generator_loss = generator_loss_fn(model, add_summaries=add_summaries)
      discriminator_loss = discriminator_loss_fn(model, add_summaries=add_summaries)

      # Gradient Penalty.
      if _use_aux_loss(gradient_penalty_weight):
        gradient_penalty_fn = tuple_losses.stargan_gradient_penalty_wrapper(
            losses_wargs.wasserstein_gradient_penalty)
        discriminator_loss += gradient_penalty_fn(
            model,
            epsilon=gradient_penalty_epsilon,
            target=gradient_penalty_target,
            one_sided=gradient_penalty_one_sided,
            add_summaries=add_summaries) * gradient_penalty_weight

      # Self-consistency Loss.
      if self_consistency_loss_weight >= 0.0:
          self_consistency_loss = self_consistency_loss_fn(model.input_data, model.generated_data)
          generator_loss += self_consistency_loss * self_consistency_loss_weight
          if add_summaries:
              tf.compat.v1.summary.scalar('self_consistency_loss', self_consistency_loss)

      # Reconstruction Loss.
      if reconstruction_loss_weight >= 0.0:
          reconstruction_loss = reconstruction_loss_fn(model.input_data, model.reconstructed_data)
          generator_loss += reconstruction_loss * reconstruction_loss_weight
          if add_summaries:
            tf.compat.v1.summary.scalar('reconstruction_loss', reconstruction_loss)

      # Classification Loss.
      generator_loss += _classification_loss_helper(
          true_labels=model.generated_data_domain_target,
          predict_logits=model.discriminator_generated_data_domain_predication,
          scope_name='generator_classification_loss') * classification_loss_weight
      discriminator_loss += _classification_loss_helper(
          true_labels=model.input_data_domain_label,
          predict_logits=model.discriminator_input_data_domain_predication,
          scope_name='discriminator_classification_loss'
      ) * classification_loss_weight

      return namedtuples.GANLoss(generator_loss, discriminator_loss)

    return stargan_loss

def train(hparams, override_generator_fn=None, override_discriminator_fn=None):
  """Trains a StarGAN.

  Args:
    hparams: An HParams instance containing the hyperparameters for training.
    override_generator_fn: A generator function that overrides the default one.
    override_discriminator_fn: A discriminator function that overrides the
      default one.
  """
  # Create directories if not exist.
  if not tf.io.gfile.exists(hparams.output_dir):
    tf.io.gfile.makedirs(hparams.output_dir)

  with open(hparams.output_dir + 'train_result.json', 'w') as fp:
    json.dump(hparams._asdict(), fp, indent=4)
    
  # Make sure steps integers are consistent.
  if hparams.max_number_of_steps % hparams.steps_per_eval != 0:
    raise ValueError('`max_number_of_steps` must be divisible by '
                     '`steps_per_eval`.')

  # Create optimizers.
  gen_opt, dis_opt = _get_optimizer(hparams.generator_lr,
                                    hparams.discriminator_lr,
                                    hparams.adam_beta1, hparams.adam_beta2)

  # Create estimator.
  if hparams.cls_model and hparams.cls_checkpoint:
    raise Exception('Can only assign one parameter between hparams.cls_model and hparams.cls_checkpoint')

  if hparams.cls_model:
    print("[!!!!] LOAD custom classification model in discriminator.")

    network_discriminator = network.CustomKerasDiscriminator(hparams.cls_model + '/base_model.h5')
    # network_discriminator = network.custom_keras_discriminator(hparams.cls_model)
    # tf.keras.estimator.model_to_estimator(keras_model_path=hparams.cls_model, model_dir='/tmp/temp_checkpoint/')
  elif hparams.cls_checkpoint:
    network_discriminator = network.custom_tf_discriminator()
  else:
    network_discriminator = network.discriminator

  stargan_estimator = tfgan.estimator.StarGANEstimator(
      model_dir= hparams.output_dir + "checkpoints/",
      generator_fn=override_generator_fn or network.generator,
      discriminator_fn=override_discriminator_fn or network_discriminator,
      # loss_fn=tfgan.stargan_loss,
      loss_fn=_get_stargan_loss(reconstruction_loss_weight=hparams.reconstruction_loss_weight,
                                self_consistency_loss_weight=hparams.self_consistency_loss_weight,
                                classification_loss_weight=hparams.classification_loss_weight),
      generator_optimizer=gen_opt,
      discriminator_optimizer=dis_opt,
      get_hooks_fn=tfgan.get_sequential_train_hooks(
          _define_train_step(hparams.gen_disc_step_ratio)),
      add_summaries=tfgan.estimator.SummaryType.IMAGES,
      config=tf.estimator.RunConfig(save_checkpoints_steps=hparams.save_checkpoints_steps,
                                    keep_checkpoint_max=hparams.keep_checkpoint_max),
      cls_model=hparams.cls_model,
      cls_checkpoint=hparams.cls_checkpoint
  )

  # Get input function for training and test images.
  if (hparams.tfdata_source):
    print("[**] load train dataset: tensorflow dataset: {x}".format(x=hparams.tfdata_source))
    train_input_fn = lambda: data_provider.provide_data(  # pylint:disable=g-long-lambda
      split='train',
      batch_size=hparams.batch_size,
      patch_size=hparams.patch_size,
      num_parallel_calls=None,
      shuffle=True,
      tfds_name=hparams.tfdata_source,
      domains=tuple(hparams.tfdata_source_domains.split(",")),
      download=eval(hparams.download),
      data_dir=hparams.data_dir)

    if hparams.tfdata_source.startswith('cycle_gan'):
        test_images_np = data_provider.provide_cyclegan_test_set(hparams.patch_size)
        num_domains = 2
    elif hparams.tfdata_source == 'celeb_a':
        test_images_np = data_provider.provide_celeba_test_set(hparams.patch_size,
                                                               download=eval(hparams.download),
                                                               data_dir=hparams.data_dir)
        num_domains = len(test_images_np)
    else:
        test_images_np, num_domains = data_provider.provide_categorized_test_set(hparams.tfdata_source,
                                                                    hparams.patch_size,
                                                                    download=eval(hparams.download),
                                                                    data_dir=hparams.data_dir)


  else:
    train_input_fn = None
    test_images_np = None
    num_domains = None
    raise Exception("TODO: support external data souce.")
    
  filename_str = os.path.join(hparams.output_dir, 'summary_image_%i.png')

  # Periodically train and write prediction output to disk.
  cur_step = 0
  while cur_step < hparams.max_number_of_steps:
    cur_step += hparams.steps_per_eval
    print("current step: {cur_step} /{max_step}".format(cur_step=cur_step, max_step=hparams.max_number_of_steps))
    stargan_estimator.train(train_input_fn, steps=cur_step)
    summary_img = _get_summary_image(stargan_estimator, test_images_np, num_domains)
    with tf.io.gfile.GFile(filename_str % cur_step, 'w') as f:
        # Handle single-channel images
        if summary_img.shape[2] == 1:
            summary_img = np.repeat(summary_img, 3, axis=2)
        PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')
