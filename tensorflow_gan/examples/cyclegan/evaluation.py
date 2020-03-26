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
""" reference: inference_demo.py """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from argparse import Namespace
from absl import app
from absl import flags
import numpy as np
import PIL
import tensorflow as tf
from tensorflow_gan.examples.cyclegan import data_provider
from tensorflow_gan.examples.cyclegan import networks
import json

def _make_dir_if_not_exists(dir_path):
    """Make a directory if it does not exist."""
    if not tf.io.gfile.exists(dir_path):
        tf.io.gfile.makedirs(dir_path)

def _file_output_path(dir_path, input_file_path):
    """Create output path for an individual file."""
    return os.path.join(dir_path, os.path.basename(input_file_path))

def make_inference_graph(model_name, patch_dim):
    """Build the inference graph for either the X2Y or Y2X GAN.

    Args:
      model_name: The var scope name 'ModelX2Y' or 'ModelY2X'.
      patch_dim: An integer size of patches to feed to the generator.

    Returns:
      Tuple of (input_placeholder, generated_tensor).
    """
    input_hwc_pl = tf.compat.v1.placeholder(tf.float32, [None, None, 3])

    # Expand HWC to NHWC
    images_x = tf.expand_dims(
        data_provider.full_image_to_patch(input_hwc_pl, patch_dim), 0)

    with tf.compat.v1.variable_scope(model_name):
        with tf.compat.v1.variable_scope('Generator'):
            generated = networks.generator(images_x)
    return input_hwc_pl, generated

def export(sess, input_pl, output_tensor, input_file_pattern, output_dir):
    """Exports inference outputs to an output directory.

    Args:
      sess: tf.Session with variables already loaded.
      input_pl: tf.Placeholder for input (HWC format).
      output_tensor: Tensor for generated outut images.
      input_file_pattern: Glob file pattern for input images.
      output_dir: Output directory.
    """
    if output_dir:
        _make_dir_if_not_exists(output_dir)

    result = {}
    if input_file_pattern:
        for file_path in tf.io.gfile.glob(input_file_pattern):
            filename = file_path.split("/")[-1]
            # Grab a single image and run it through inference
            input_np = np.asarray(PIL.Image.open(file_path))
            output_np = sess.run(output_tensor, feed_dict={input_pl: input_np})
            image_np = data_provider.undo_normalize_image(output_np)
            generated_filename = "generated_from_" + filename
            result[filename] = image_np
            if (output_dir):
                output_path = _file_output_path(output_dir, generated_filename)
                print("output: {output_path}".format(output_path=output_path))
                PIL.Image.fromarray(image_np).save(output_path)
        return result

def check_input_info(input_json):
    if (not 'checkpoint_path' in input_json): raise Exception("Please provide model checkpoint.")
    
    if ("x2y_images_dir" in input_json):
        input_json.x2y_images_dir = input_json.output_dir + input_json.x2y_images_dir
        
    if ("y2x_images_dir" in input_json):
        input_json.y2x_images_dir = input_json.output_dir + input_json.y2x_images_dir
    
    if ("x_images" in input_json and not "x2y_images_dir" in input_json):
        raise Exception("Please provide output dir for y: x-->y.")

    if ("y_images" in input_json and not "y2x_images_dir" in input_json):
        raise Exception("Please provide output dir for x: y-->x.")
      
def main(inputs):
    
    if not tf.io.gfile.exists(inputs.output_dir):
        tf.io.gfile.makedirs(inputs.output_dir)
        
    with open(inputs.output_dir + 'evaluation.json', 'w') as fp:
        json.dump(inputs.__dict__, fp, indent=4)
        
    print("evluation result: {x}".format(x=inputs.output_dir))
    check_input_info(inputs)
    images_x_hwc_pl, generated_y = make_inference_graph('ModelX2Y', inputs.patch_size)
    images_y_hwc_pl, generated_x = make_inference_graph('ModelY2X', inputs.patch_size)

    # Restore all the variables that were saved in the checkpoint.
    saver = tf.compat.v1.train.Saver()
    generated_image_x, generated_image_y = None, None
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, inputs.checkpoint_path)
        if ("x_images" in inputs):
            print("Convert x to y ...")
            generated_images_y = export(sess, images_x_hwc_pl, generated_y, inputs.x_images, inputs.x2y_images_dir)
        if ("y_images" in inputs):
            print("Convert y to x ...")          
            generated_images_x = export(sess, images_y_hwc_pl, generated_x, inputs.y_images,inputs.y2x_images_dir)
    return generated_image_x, generated_image_y
  
if __name__ == '__main__':
    # python inference_demo.py --checkpoint_path=./model_ckpts/cyclegan/model.ckpt-464267 --image_set_x_glob=./testdata/*.jpg --generated_y_dir=./testdata/tmp/generated_y/ --patch_dim=64
    input_json = {
        # "checkpoint_path": "./model_ckpts/cyclegan/model.ckpt-464267",
        "checkpoint_path": "/Users/ptien/tfds-download/models_ckpts/tfgan_logdir/cyclegan/model.ckpt-500000",
        "train_data_sourec": "apple2orange",
        # "x_images": "./testdata/*.jpg",
        "x_images": "/Users/ptien/tfds-download/apple2orange/testA/*.jpg", # "n07740461_10011.jpg",
        # "x2y_images_dir": "./testdata/tmp2/generated_y/",  # x2y: transform x to y.
        "output_dir": "/Users/ptien/tfds-download/apple2orange/experiment/",
        "x2y_images_dir": "generated_y/", # sub folder under output_dir
        # "y_images": "./testdata/*.jpg",
        # "y2x_images_dir": "./testdata/tmp/generated_x/",  # y2x: transform y to x.
        "patch_size": 64,
    }
    if (type(input_json) !=Namespace):
        input_json = Namespace(**input_json)

    generated_image_x, generated_image_y = main(input_json)
