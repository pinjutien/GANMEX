import tensorflow.keras as keras

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten,
                                     GlobalAveragePooling2D, GlobalMaxPool2D, Dense, Activation)


def get_model(input_shape,
              base_model='vgg16',
              additional_conv_layers=0,
              global_average_pooling=False,
              global_max_pooling=False,
              conv_batch_normalization=True,
              conv_dropout=True,
              dense_batch_normalization=True,
              dense_dropout=True,
              dense_sizes=[1024],
              ):

    if global_average_pooling and global_max_pooling:
        raise Exception('Can only choose one between global_average_pooling and global_max_pooling')

    # load model without classifier layers
    if base_model.lower() == 'vgg16':
        pre_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise Exception('Unsupported base model ' + base_model)
    output = pre_model.output

    # add convolutional layers
    while additional_conv_layers:
        if output.shape[1] > 1 and output.shape[2] > 1:
            output = Conv2D(output.shape[-1] * 2, (3, 3), strides=(1, 1), activation='relu', padding='same')(output)
            if conv_batch_normalization:
                output = BatchNormalization()(output)
            output = MaxPooling2D(pool_size=(2, 2))(output)
            if conv_dropout:
                output = Dropout(0.25)(output)
        else:
            print("Can't add convolutional layer after output size: " + str(pre_model.output.shape))
        additional_conv_layers -= 1

    # add a global pooling layer
    if global_average_pooling:
        output = GlobalAveragePooling2D()(output)
    elif global_max_pooling:
        output = GlobalMaxPool2D()(output)

    output = Flatten()(output)

    # add dense layers
    for ds in dense_sizes:
        output = Dense(ds, activation='relu')(output)
        if dense_batch_normalization:
            output = BatchNormalization()(output)
        if dense_dropout:
            output = Dropout(0.25)(output)

    output = Dense(2, activation=None)(output)
    output = Activation('softmax')(output)

    # define new model
    model = Model(inputs=pre_model.input, outputs=output)

    for layer in model.layers[:19]:
        layer.trainable = False

    return model
