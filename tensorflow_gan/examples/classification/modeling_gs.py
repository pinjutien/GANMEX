#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd 
import tensorflow as tf
import tensorflow_datasets as tfds
# import tensorflow.keras as keras
# import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications import (VGG16, MobileNet, MobileNetV2, InceptionResNetV2,
                                           InceptionV3, DenseNet121, DenseNet169, DenseNet201,
                                           NASNetLarge, NASNetMobile, ResNet50, VGG19, Xception)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import itertools
from time import time

# from keras.applications import VGG16
# from keras.models import Model
# from keras.layers import Dense
# from keras.layers import Flatten

use_cpu = True
if (use_cpu):
    print("Use CPU only")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_data(data_path):
    filenames = []
    categories = []
    label_map = {
        'trainA': 'horse',
        'trainB': 'zebra',
    }
    for folder in os.listdir(data_path):
        if folder in label_map.keys():
            for file in os.listdir(data_path + '/' + folder):
                categories.append(label_map[folder])
                filenames.append(folder + '/' + file)

    table = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    return table, data_path


def data_gen(data_path, IMAGE_SIZE, batch_size):
    df, data_path = get_data(data_path)
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    evaluation_datagen = ImageDataGenerator(rescale=1./255)    

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        data_path, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        a=batch_size
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        data_path, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    evaluation_generator = evaluation_datagen.flow_from_dataframe(
        validate_df, 
        data_path, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=1,
        shuffle=False)
    return train_generator, validation_generator, evaluation_generator, total_train, total_validate




def performance(model, data_gen, top_k, output):
    if (output):
        os.makedirs(output, exist_ok=True)
    assert data_gen.batch_size ==1, "Require batch size = 1."
    steps = data_gen.n // data_gen.batch_size
    prob_preds = model.predict_generator(data_gen, steps= steps)
    y_labels_index = data_gen.classes
    check_arr = []
    for i in range(len(prob_preds)):
        prob_row = prob_preds[i]
        y_index = y_labels_index[i]
        top_k_pred = prob_row.argsort()[-top_k:]
        check_arr += [y_index in top_k_pred]

    pred_idxs = np.argmax(prob_preds, axis=1)
    label_to_index = (data_gen.class_indices)
    labels = dict((v,k) for k,v in label_to_index.items())
    pred_output = [ labels[i] for i in pred_idxs]
    y = [ labels[i] for i in y_labels_index]
    filenames = data_gen.filenames
    results=pd.DataFrame({"Filename":filenames,
                          "Predictions":pred_output,
                          "label": y,
                          # "in_top_k": check_arr
    })
    results["match"] = (results["Predictions"].values == results["label"].values)
    # top_k_accuracy = results["in_top_k"].mean()
    print("Test score: {x}".format(x=results["match"].mean()))
    # print("Top k score: {x}".format(x=top_k_accuracy))
    results.to_csv(os.path.join(output, "performance.csv"))


def modeling(data_gens, pretrain_model, optimizer, batch_size, epochs, ckpt_path,
             IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
    os.makedirs(ckpt_path, exist_ok=True)    
    train_generator, validation_generator, evaluation_generator, total_train, total_validate = data_gens
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    # load model without classifier layers
    # pre_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    pre_model = pretrain_model(weights='imagenet', include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    pre_model.trainable=False
    # add new classifier layers
    flat1 = Flatten()(pre_model.output)
    class1 = Dense(1024, activation='relu')(flat1)
    output = Dense(2, activation='softmax')(class1)

    # define new model
    model = Model(inputs=pre_model.input, outputs=output)

    # for layer in model.layers[:19]:
    #     layer.trainable = False

    base_learning_rate = 0.0001

    # model.summary() # show model summary
    model.compile(optimizer=optimizer(base_learning_rate), # tf.keras.optimizers.Adadelta(base_learning_rate),# optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    mdlckpt = ModelCheckpoint(os.path.join(ckpt_path,'best-model.h5'),
                              # os.path.join(ckpt_path,'model-{epoch:03d}-{val_acc:03f}.h5'),
                              save_best_only=True, monitor='val_acc', mode='max')
    callbacks = [earlystop, learning_rate_reduction, mdlckpt]
    history = model.fit_generator(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=max(150, total_validate//batch_size),
        steps_per_epoch=max(150,total_train//batch_size),
        callbacks=callbacks
    )

    model.save_weights(os.path.join(ckpt_path, "{model_name}_weights_keras.h5".format(model_name=model_name)))
    model.save(os.path.join(ckpt_path,"{model_name}_keras.h5".format(model_name=model_name)))
    output1 = os.path.join(ckpt_path, "performance")
    performance(model, evaluation_generator, 1, output1)


def generate_hyperparameter(params_dict):
    # generate all combination
    keys = params_dict.keys()
    values = (params_dict[key] for key in keys)
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return combinations
    
if __name__ == '__main__':
    IMAGE_WIDTH=256
    IMAGE_HEIGHT=256
    IMAGE_CHANNELS=3
    data_path = '/home/ptien/tfds-download/apple2orange/'
    # ckpt_path = "./test_model/"
    # model_name = "horse2zebra"
    epochs=200
    batch_size=32
    model_dict = {
        "VGG16": VGG16,
        "MobileNet": MobileNet,
        "MobileNetV2": MobileNetV2,
        "InceptionResNetV2": InceptionResNetV2,
        "InceptionV3": InceptionV3,
        "DenseNet121": DenseNet121,
        "DenseNet169": DenseNet169,
        "DenseNet201": DenseNet201,
        "NASNetLarge": NASNetLarge,
        "NASNetMobile": NASNetMobile,
        "ResNet50": ResNet50,
        "VGG19": VGG19,
        "Xception": Xception
    }
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
    experiment_confg = {
        "optimizer": ["Adadelta", "Adam"],
        "pretrain_model": ["VGG16", "MobileNet"]
    }
    experiments = generate_hyperparameter(experiment_confg)
    num_exp = len(experiments)
    print("number of experiments: {x}".format(x=num_exp))
    output = "/home/ptien/temp/experiments/"
    for i in range(num_exp):
        t0 = time()
        exp_config = experiments[i]
        IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
        data_gens = data_gen(data_path, IMAGE_SIZE, batch_size)
        model_name = exp_config['pretrain_model']
        opt_name = exp_config['optimizer']
        pretrain_model = model_dict[model_name]
        optimizer = optimizer_dict[opt_name]
        exp_name = str(i)+ "_" + model_name + "_" + opt_name
        print(exp_name)
        ckpt_path = os.path.join(output, exp_name)
        modeling(data_gens, pretrain_model, optimizer, batch_size, epochs, ckpt_path,
                 IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
        time_elpase = (time() -t0)/60.0
        print("time elpase: {x}".format(x=time_elpase))
        del data_gens
