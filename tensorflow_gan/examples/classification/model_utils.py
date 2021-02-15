import tensorflow.keras as keras

import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNet
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Conv2D, BatchNormalization, Dropout, Flatten,
                                     GlobalAveragePooling2D, AveragePooling2D, Dense, Activation)


def get_optimizer(args):
    optimizer_dict = {
        "Adadelta": tf.keras.optimizers.Adadelta,
        "Adagrad": tf.keras.optimizers.Adagrad,
        "Adam": tf.keras.optimizers.Adam,
        "Adamax": tf.keras.optimizers.Adamax,
        "Ftrl": tf.keras.optimizers.Ftrl,
        "SGD": tf.keras.optimizers.SGD,
        "RMSprop": tf.keras.optimizers.RMSprop,
        "Nadam": tf.keras.optimizers.Nadam
    }
    optimizer_dict = {k.lower(): v for k, v in optimizer_dict.items()}
    optimizer = optimizer_dict[args.optimizer.lower()](learning_rate=args.base_learning_rate)

    return optimizer


def get_base_model(base_model_name, input_shape):
    if base_model_name.lower() == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name.lower() == 'mobilenet':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise Exception('Unsupported base model ' + base_model_name)

    return base_model


def get_model(input_shape,
              base_model_name='vgg16',
              additional_conv_layers=0,
              global_average_pooling=False,
              conv_batch_normalization=True,
              conv_dropout=True,
              dense_batch_normalization=True,
              dense_dropout=True,
              dense_sizes=[1024],
              num_classes=2
              ):

    # load model without classifier layers
    base_model = get_base_model(base_model_name, input_shape)

    base_model_size = len(base_model.layers)
    output = base_model.output

    # add convolutional layers
    while additional_conv_layers:
        if output.shape[1] > 1 and output.shape[2] > 1:
            output = Conv2D(output.shape[-1] * 2, (3, 3), strides=(1, 1), activation='relu', padding='same')(output)
            if conv_batch_normalization:
                output = BatchNormalization()(output)
            output = AveragePooling2D(pool_size=(2, 2))(output)
            if conv_dropout:
                output = Dropout(0.25)(output)
        else:
            print("Can't add convolutional layer after output size: " + str(base_model.output.shape))
        additional_conv_layers -= 1

    # add a global pooling layer
    if global_average_pooling:
        output = GlobalAveragePooling2D()(output)

    output = Flatten()(output)

    # add dense layers
    for ds in dense_sizes:
        output = Dense(ds, activation='relu')(output)
        if dense_batch_normalization:
            output = BatchNormalization()(output)
        if dense_dropout:
            output = Dropout(0.25)(output)

    output = Dense(num_classes, activation=None)(output)
    output = Activation('softmax')(output)

    # define new model
    model = Model(inputs=base_model.input, outputs=output)

    for layer in model.layers[:base_model_size]:
        layer.trainable = False

    return model


def get_cyclegan_model(input_shape, base_model_name, num_classes=2):
    base_model = get_base_model(base_model_name, input_shape)

    base_model_size = len(base_model.layers)
    output = base_model.output

    output = AveragePooling2D(pool_size=(2, 2))(output)
    output = Flatten()(output)
    output = Dense(1024, activation='relu')(output)
    output = Dropout(0.5)(output)

    output = Dense(num_classes, activation=None)(output)
    output = Activation('softmax')(output)

    # define new model
    model = Model(inputs=base_model.input, outputs=output)

    for layer in model.layers[:base_model_size]:
        layer.trainable = False

    return model


def get_mnist_model(input_shape, num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=6, strides=2, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=6, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def get_svhn_model(input_shape, num_classes=10):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=5, strides=2, activation='relu'))
    model.add(Conv2D(128, kernel_size=4, strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


def get_cifar10_model(input_shape, num_classes=10):
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(AveragePooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model
