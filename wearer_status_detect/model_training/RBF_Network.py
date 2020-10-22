"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2020/10/22 1:46:38
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : RBF_Network.py
# @Software: PyCharm
-------------------------------------
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import model_selection

# use gpu 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
LR = 0.0001  # learning rate
EPOCHS = 100
BATCH_SIZE = 100


def data_preprocessing(feature, label):
    """

    process the data in order to train the madel

    Args:
        feature: the data with six dimension
        label: the label of the data

    Returns:
        feature: the normalized data
        label_list: the one-hot label

    """

    feature = feature.reshape(-1, 6)

    label_list = []
    for line in label:
        temp = [0, 0, 0, 0]
        temp[int(line) - 1] = 1
        label_list.append(temp)

    label_list = np.array(label_list)
    label_list = label_list.reshape(-1, 4)
    result_view = pd.DataFrame(label_list)
    result_view.to_csv('../label_test/result_view.txt', sep='\t', index=False)

    return feature, label_list


def train_data(X_train, y_train, X_test, y_test):
    """

    train the BP model

    Args:
        the data to train and test

    Returns:
        model: the trained model
        history: the log of training

    """
    model = tf.keras.Sequential()

    # rbflayer = RBFLayer(9,
    #                     initializer=InitCentersRandom(X_train),
    #                     betas=2.0,
    #                     input_shape=(6,))
    # model.add(rbflayer)

    model.add(tf.keras.layers.Dense(100,
                                    kernel_regularizer=tf.keras.regularizers.l2(
                                        0.001), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(50,
                                    kernel_regularizer=tf.keras.regularizers.l2(
                                        0.001), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(4,
                                    kernel_regularizer=tf.keras.regularizers.l2(
                                        0.001), activation='sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    # model = tf.keras.models.load_model("../model/RBF_QG.h5",
    #                                    custom_objects={'RBFLayer': RBFLayer})

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    tf.keras.models.save_model(model=model, filepath="../model/RBF_QG.h5")

    return model, history


def show_pic(history):
    """show the loss descend

    Args:
        the training log

    Returns:
        None

    """
    # show the history of loss function
    history.history.keys()
    plt.plot(history.epoch, history.history.get('loss'))
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("loss-descend")
    plt.show()


if __name__ == '__main__':
    dataSet = pd.read_csv('../data/data_generated_by_QG/data_by_QG.csv')
    dataSet = dataSet[~dataSet['feature_1'].isin([0.0])]
    dataSet = dataSet[~dataSet['feature_2'].isin([0.0])]
    dataSet = dataSet[~dataSet['feature_3'].isin([0.0])]
    dataSet = dataSet[~dataSet['feature_4'].isin([0.0])]
    dataSet = dataSet[~dataSet['feature_5'].isin([0.0])]
    dataSet = dataSet[~dataSet['feature_6'].isin([0.0])]

    X = np.array(dataSet.iloc[:, 0:6])
    y = np.array(dataSet.loc[:, 'label'])

    X, y = data_preprocessing(X, y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                        shuffle=False,
                                                                        test_size=0.3)

    # train model
    model, history = train_data(X_train, y_train, X_test, y_test)
    show_pic(history)
