import tensorflow as tf


def load_data(input_file_pattern):
    x_and_y = []
    for file_path in tf.io.gfile.glob(input_file_pattern):
        x_and_y += [(file_path, labels)]
    return tuple(x_and_y)

def get_label(file_path):
    default_label = {"black_hair": 0,
                     "blond_hair": 0,
                     "brown_hair": 1}
    return default_label
    
