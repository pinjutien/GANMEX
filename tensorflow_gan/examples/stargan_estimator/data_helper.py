import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_gan.examples.cyclegan import data_provider as cyclegan_dp
import numpy as np
AUTOTUNE = tf.data.experimental.AUTOTUNE

def load_data(input_file_pattern):
    list_ds = tf.data.Dataset.list_files(input_file_pattern)
    return list_ds
    # x = []
    # for file_path in tf.io.gfile.glob(input_file_pattern):
    #     x += [file_path]
    # return tuple(x)

def get_label(file_path, label=None):
    if (label is None):
        default_label = {"black_hair": 0,
                         "blond_hair": 0,
                         "brown_hair": 1}
        return default_label
    else:
        return label

def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    # return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    return img

def process_path(file_path, patch_size):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = cyclegan_dp.full_image_to_patch(img, patch_size)
    return img, label

def load_custom_data_v1(input_file_pattern, patch_size):
    list_ds = tf.data.Dataset.list_files(input_file_pattern, AUTOTUNE)
    ds = list_ds.map(process_path)
    ds_np = tfds.as_numpy(ds)
    # Get one image of each hair type.
    images = []
    labels = []
    while len(images) < 3:
        # elem = next(ds_np)
        (image_x, labels_dict) = next(ds_np)
        cur_lbl = [labels_dict['Black_Hair'.lower()],
                   labels_dict['Blond_Hair'.lower()],
                   labels_dict['Brown_Hair'.lower()]]
        if cur_lbl not in labels:
            images.append(image_x)
            labels.append(cur_lbl)
    images = np.array(images, dtype=np.float32)
    assert images.dtype == np.float32
    assert np.max(np.abs(images)) <= 1.0
    assert images.shape == (3, patch_size, patch_size, 3)
    return images

def load_custom_data(input_file_pattern, patch_size, domains=['Black_Hair','Blond_Hair','Brown_Hair']):
    # Get one image of each hair type.
    images = []
    images_name = []
    labels = []
    for file_path in tf.io.gfile.glob(input_file_pattern):
        image_x, labels_dict = process_path(file_path, patch_size)
        # cur_lbl = [labels_dict['Black_Hair'.lower()],
        #            labels_dict['Blond_Hair'.lower()],
        #            labels_dict['Brown_Hair'.lower()]]
        # if cur_lbl not in labels:
        #     images += [image_x]
        #     labels += [cur_lbl]
        cur_lbl = [ labels_dict[dm.lower()] for dm in domains]
        images += [tfds.as_numpy(image_x)]
        images_name += [file_path]
        labels += [cur_lbl]
    return np.array(images), np.array(images_name), np.array(labels)


if __name__ == '__main__':
    patch_size = 128
    input_file_pattern = "./testdata/*.png"
    domains=['Black_Hair','Blond_Hair','Brown_Hair']
    # images = provide_custom_data(input_file_pattern, patch_size)
    images, images_name, labels = load_custom_data(input_file_pattern, patch_size, domains)
    print("end")
