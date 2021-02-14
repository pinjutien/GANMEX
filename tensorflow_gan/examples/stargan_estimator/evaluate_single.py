from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import sys
# sys.path.append("../../..")

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan

from tensorflow_gan.examples.stargan import network
from tensorflow_gan.examples.cyclegan import data_provider as cyclegan_dp
from tensorflow_gan.python.estimator import stargan_estimator

from tensorflow_gan.examples.stargan_estimator import data_provider


# # Set the PREDICT_FLAG for debugging
# stargan_estimator.PREDICT_FLAG = True

# tfdata_source = 'svhn_cropped'
# patch_size = 32
# n_classes = 10

# tfdata_source = 'mnist'
# patch_size = 28
# n_classes = 10

tfdata_source = 'cifar10'
patch_size = 32
n_classes = 10

use_color_labels = 0


def _get_summary_image(estimator, checkpoint_path, test_images_np, num_domains):
  """Returns a numpy image of the generate on the test images."""

  img_rows = []
  for img_np in test_images_np:

    def test_input_fn():
      dataset_imgs = [img_np] * num_domains  # pylint:disable=cell-var-from-loop
      dataset_lbls = [tf.one_hot([d], num_domains) for d in range(num_domains)]

      # Make into a dataset.
      dataset_imgs = np.stack(dataset_imgs)
      dataset_imgs = np.expand_dims(dataset_imgs, 1)
      dataset_lbls = tf.stack(dataset_lbls)
      unused_tensor = tf.zeros(num_domains)
      return tf.data.Dataset.from_tensor_slices(((dataset_imgs, dataset_lbls),
                                                 unused_tensor))

    prediction_iterable = estimator.predict(test_input_fn, checkpoint_path=checkpoint_path)
    predictions = [next(prediction_iterable) for _ in range(num_domains)]
    transform_row = np.concatenate([img_np] + predictions, 1)
    img_rows.append(transform_row)

  all_rows = np.concatenate(img_rows, 0)
  # Normalize` [-1, 1] to [0, 1].
  normalized_summary = (all_rows + 1.0) / 2.0
  return normalized_summary



if __name__ == '__main__':

    # discriminator_fn = network.discriminator
    discriminator_fn = network.CustomKerasDiscriminator('/home/ec2-user/gan/gan_submission_1/svhn/svhn_v2/base_model.h5')

    stargan_estimator = tfgan.estimator.StarGANEstimator(
        model_dir=None,
        generator_fn=network.generator,
        discriminator_fn=discriminator_fn,
        loss_fn=tfgan.stargan_loss,
        add_summaries=tfgan.estimator.SummaryType.IMAGES
    )

    checkpoint_dir = '/tmp/mnist_vx_cw100_sw20_rw10/stargan_estimator/out/checkpoints/'
    cur_step = 50000

    if checkpoint_dir[-1] != '/':
        checkpoint_dir += '/'
    checkpoint_path = checkpoint_dir + 'model.ckpt-%s' % cur_step

    for label in range(n_classes):
        test_images_np, num_domains = data_provider.provide_categorized_test_set(tfdata_source,
                                                                                 patch_size,
                                                                                 color_labeled=use_color_labels,
                                                                                 num_images=10,
                                                                                 filtered_label=label)

        summary_img = _get_summary_image(stargan_estimator, checkpoint_path, test_images_np, num_domains)

        # Handle single-channel images
        if summary_img.shape[2] == 1:
            summary_img = np.repeat(summary_img, 3, axis=2)

        with tf.io.gfile.GFile(checkpoint_dir + 'sample_image_%d_l%d.png' % (cur_step, label), 'w') as f:
            PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')
