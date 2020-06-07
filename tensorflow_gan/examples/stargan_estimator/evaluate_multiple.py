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
from glob import glob
from tensorflow_gan.examples.stargan import network
from tensorflow_gan.examples.cyclegan import data_provider as cyclegan_dp
from tensorflow_gan.examples.cyclegan.data_provider import load_data_from


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


def get_checkpoint_struct(checkpoint_dir, increments=[2000, 10000]):
    files = os.listdir(checkpoint_dir)
    data = {}

    for file in files:
        for suffix in ['.data-00000-of-00001', '.index', '.meta']:
            if file.endswith(suffix):
                checkpoint = file[:-len(suffix)]
                if checkpoint not in data:
                    data[checkpoint] = []
                data[checkpoint].append(suffix)

    prefix = 'model.ckpt-'
    checkpoints = [k for k, v in data.items() if tuple(sorted(v)) == ('.data-00000-of-00001', '.index', '.meta')
                   and k.startswith(prefix)]

    checkpoint_map = {int(ckpt[len(prefix):]): (checkpoint_dir + ckpt) for ckpt in checkpoints}

    checkpoint_struct = {}
    checkpoint_struct['all'] = [checkpoint_map[k] for k in sorted(checkpoint_map.keys())]

    for inc in increments:
        checkpoint_struct[str(inc)] = [checkpoint_map[k] for k in sorted(checkpoint_map.keys()) if k % inc == 0]

    return checkpoint_struct


def make_summary_images(checkpoint_dir, checkpoint_struct, dataset_name, num_examples=10):

    ds = tfds.load(dataset_name)
    examples_apples = list(tfds.as_numpy(ds['testA'].take(num_examples)))
    examples_oranges = list(tfds.as_numpy(ds['testB'].take(num_examples)))
    input_apples = [tfds.as_numpy(cyclegan_dp.full_image_to_patch(x['image'], 128)).astype('float32') for x in
                    examples_apples]
    input_oranges = [tfds.as_numpy(cyclegan_dp.full_image_to_patch(x['image'], 128)).astype('float32') for x in
                     examples_oranges]

    stargan_estimator = tfgan.estimator.StarGANEstimator(
        model_dir=None,
        generator_fn=network.generator,
        discriminator_fn=network.discriminator,
        loss_fn=tfgan.stargan_loss,
        add_summaries=tfgan.estimator.SummaryType.IMAGES
    )

    summary_apples, summary_oranges = {}, {}
    for checkpoint_path in checkpoint_struct['all']:
        summary_apples[checkpoint_path] = translate_images(stargan_estimator, input_apples, 1, checkpoint_path, 2)
        summary_oranges[checkpoint_path] = translate_images(stargan_estimator, input_oranges, 0, checkpoint_path, 2)

    for key, struct in checkpoint_struct.items():
        all_rows_apple, all_rows_orange = [], []

        for ind in range(num_examples):
            image_apple = (input_apples[ind] + 1.0) / 2.0
            image_orange = (input_oranges[ind] + 1.0) / 2.0
            row_apple = np.concatenate([image_apple] + [summary_apples[x][ind] for x in struct], 1)
            row_orange = np.concatenate([image_orange] + [summary_oranges[x][ind] for x in struct], 1)

            all_rows_apple.append(row_apple)
            all_rows_orange.append(row_orange)

        summary_apple = np.concatenate(all_rows_apple, 0)
        summary_orange = np.concatenate(all_rows_orange, 0)

        with tf.io.gfile.GFile(checkpoint_dir + 'summary_apple_' + key + '.png', 'w') as f:
            PIL.Image.fromarray((255 * summary_apple).astype(np.uint8)).save(f, 'PNG')
        with tf.io.gfile.GFile(checkpoint_dir + 'summary_orange_' + key + '.png', 'w') as f:
            PIL.Image.fromarray((255 * summary_orange).astype(np.uint8)).save(f, 'PNG')



# checkpoint_path_pattern = '/Users/shengms/Code/gan_checkpoints/stargan_est_a2o_rw10/model.ckpt-%d'
# checkpoint_numbers = list(range(10000, 140001, 10000))
#
# with tf.io.gfile.GFile(filename_str % cur_step, 'w') as f:
#     PIL.Image.fromarray((255 * summary_img).astype(np.uint8)).save(f, 'PNG')


if __name__ == '__main__':
    # checkpoint_dir = '/tmp/tfgan_logdir/stargan_estimator/out/checkpoints/'
    checkpoint_dir = "/Users/pin-jutien/tfds-download/models_ckpts/stargan_est_glr2m5_gd1/"
    if checkpoint_dir[-1] != '/':
        checkpoint_dir += '/'
    data_dir = "/Users/pin-jutien/tfds-download/apple2orange/"
    checkpoint_struct = get_checkpoint_struct(checkpoint_dir, increments=[10000])
    make_summary_images(checkpoint_dir, checkpoint_struct, 'cycle_gan', num_examples=10)
