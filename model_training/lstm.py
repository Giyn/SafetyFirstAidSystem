"""
-------------------------------------------------
# @Time: 2020/9/26 16:49
# @USER: 86199
# @File: csv_to_txt
# @Software: PyCharm
# @license: Copyright(C), xxxCompany
# @Author: 张平路
-------------------------------------------------
# @Attantion：
#    1、this file's function is to train the lstm model,which is abandoned after the first BPNN
-------------------------------------------------
"""
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn import model_selection
import tensorflow as tf
import os
from scipy import stats
import matplotlib.pyplot as plt

# use gpu to train the model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# hyper parameter
NUM = 1024000
DROP_OUT1 = 0.4
DROP_OUT2 = 0.1
LR = 0.001
LOAD_NAME = 'test1.h5'
SAVE_NAME = 'test2.h5'
EPOCHS = 5
BATCH_SIZE = 50
LOAD_MODEL_FLAG = 0
SAVE_MODEL_FLAG = 1  # whether save model


def normalization(data):
    """normalize the data

    Args:
        data: the data with dimsention 2
    Returns:
        the normalized data

    """
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def data_preprocess(data, label):
    """process the data in order to train the madel

    normalize the data, and change label dimension from (None,1) to (None, 7)

    Args:
        data: the data with six dimension
        label: the label of the data

    Returns:
        data:the normalized data
        y_new: the one-hot label


    """

    for i in label:
        for j in i:
            j = int(j)  # transform into int

    data = data.reshape(-1, 128, 6)

    y = []
    for i in label:
        # the mode of last one forth data
        y.append(stats.mode(i[64:])[0][0])

    # convert to one-hot code
    y_new = []
    for line in y:
        temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        temp[int(line)] = 1
        y_new.append(temp)
    y_new = np.array(y_new)
    y_new = y_new.reshape(-1, 13)

    return data, y_new


def train_data(X_train, y_train, X_test, y_test):
    """train the BP model

    Args:
        the data to train and test

    Returns:
        model： the trained BP network
        history: the log of processing

    """
    # train the model
    model = tf.keras.Sequential()

    # the hidden layer
    model.add(layers.LSTM(units=200, return_sequences=True, input_shape=(128, 6), use_bias=True, dropout=DROP_OUT1,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01), unit_forget_bias=True))
    model.add(layers.LSTM(units=52, input_shape=(128, 6), use_bias=True, dropout=DROP_OUT2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01), unit_forget_bias=True))

    # the output layer
    model.add(layers.Dense(13, activation='softmax', use_bias=True))

    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])
    if (LOAD_MODEL_FLAG):
        model = tf.keras.models.load_model(LOAD_NAME)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test), validation_freq=1, callbacks=[reduce_lr], shuffle=True,
                        workers=2)

    model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    if SAVE_MODEL_FLAG:
        model.save(filepath=SAVE_NAME)


def show_pic(history):
    """show the loss descend

    Args:
        the training log

    Returns:
        None

    """
    # show the history of loss
    history.history.keys()
    plt.plot(history.epoch, history.history.get('loss'))
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title("loss-descend")
    plt.show()


if __name__ == "__main__":
    print("----Start----")

    # read the csv
    df = pd.read_csv(r'../data/deleted_total_data.csv').drop(['x', 'id'], axis=1)
    df = df.reindex(np.random.permutation(df.index))[:NUM]  # random the data

    # data preprocess
    label = np.array(df.loc[:, 'label'])
    data = np.array(df.iloc[:, 0:-1])

    data, y_new = data_preprocess(data, label)

    # split the data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, y_new, shuffle=False, test_size=0.3)

    # train the model
    model, history = train_data(X_train, y_train, X_test, y_test)

    show_pic(history)

    print("----End------")
