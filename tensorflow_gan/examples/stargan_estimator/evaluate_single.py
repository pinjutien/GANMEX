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

stargan_estimator.PREDICT_FLAG = True

def translate_images(estimator, test_images_list, label, checkpoint_path, num_domains):
    """Returns a numpy image of the generate on the test images."""
    img_rows = []

    def test_input_fn():
        dataset_lbls = [tf.one_hot([label], num_domains)] * len(test_images_list)

        # Make into a dataset.
        dataset_imgs = np.stack(test_images_list)
        dataset_imgs = np.expand_dims(dataset_imgs, 1)
        dataset_lbls = tf.stack(dataset_lbls)
        unused_tensor = tf.zeros(len(test_images_list))
        return tf.data.Dataset.from_tensor_slices(((dataset_imgs, dataset_lbls),
                                                   unused_tensor))

    prediction_iterable = estimator.predict(test_input_fn, checkpoint_path=checkpoint_path)
    predictions = [next(prediction_iterable) for _ in range(len(test_images_list))]  # range(len(test_images_list))]
    normalized_summary = [(result + 1.0) / 2.0 for result in predictions]
    return normalized_summary


def make_summary_images(checkpoint_dir, checkpoint_file, dataset_name, num_examples=10):


    ds = tfds.load(dataset_name)
    examples_apples = list(tfds.as_numpy(ds['testA'].take(num_examples)))
    examples_oranges = list(tfds.as_numpy(ds['testB'].take(num_examples)))
    input_apples = [tfds.as_numpy(cyclegan_dp.full_image_to_patch(x['image'], 256)).astype('float32') for x in
                    examples_apples]
    input_oranges = [tfds.as_numpy(cyclegan_dp.full_image_to_patch(x['image'], 256)).astype('float32') for x in
                     examples_oranges]

    discriminator_fn = network.discriminator
    # discriminator_fn = network.CustomKerasDiscriminator('/Users/shengms/Code/gan_checkpoints/stargan_est_glr2m5_gd1_ab09_875886/model-032-0.875886.h5')

    stargan_estimator = tfgan.estimator.StarGANEstimator(
        model_dir=None,
        generator_fn=network.generator,
        discriminator_fn=discriminator_fn,
        loss_fn=tfgan.stargan_loss,
        add_summaries=tfgan.estimator.SummaryType.IMAGES
    )

    summary_apples = translate_images(stargan_estimator, input_apples, 1, checkpoint_dir + checkpoint_file, 2)
    summary_oranges = translate_images(stargan_estimator, input_oranges, 0, checkpoint_dir + checkpoint_file, 2)

    all_rows_apple, all_rows_orange = [], []

    for ind in range(num_examples):
        image_apple = (input_apples[ind] + 1.0) / 2.0
        image_orange = (input_oranges[ind] + 1.0) / 2.0
        row_apple = np.concatenate([image_apple, summary_apples[ind]], 1)
        row_orange = np.concatenate([image_orange, summary_oranges[ind]], 1)

        all_rows_apple.append(row_apple)
        all_rows_orange.append(row_orange)

    summary_apple = np.concatenate(all_rows_apple, 0)
    summary_orange = np.concatenate(all_rows_orange, 0)

    with tf.io.gfile.GFile(checkpoint_dir + 'summary_apple_' + checkpoint_file + '.png', 'w') as f:
        PIL.Image.fromarray((255 * summary_apple).astype(np.uint8)).save(f, 'PNG')
    with tf.io.gfile.GFile(checkpoint_dir + 'summary_orange_' + checkpoint_file + '.png', 'w') as f:
        PIL.Image.fromarray((255 * summary_orange).astype(np.uint8)).save(f, 'PNG')


if __name__ == '__main__':
    # checkpoint_dir = '/tmp/tfgan_logdir_875886_temp/stargan_estimator/out/checkpoints/'
    # checkpoint_file = 'model.ckpt-0'

    # checkpoint_dir = '/Users/shengms/Code/gan_checkpoints/stargan_est_glr2m5_gd1_ab09/'
    # checkpoint_file = 'model.ckpt-130000'

    checkpoint_dir = '/Users/shengms/Code/gan_checkpoints/stargan_est_share1/'
    checkpoint_file = 'model.ckpt-0'

    # checkpoint_dir = '/Users/shengms/Code/gan_checkpoints/stargan_est_875886_temp/'
    # checkpoint_file = 'model.ckpt-100'

    if checkpoint_dir[-1] != '/':
        checkpoint_dir += '/'
    make_summary_images(checkpoint_dir, checkpoint_file, 'cycle_gan', num_examples=10)
