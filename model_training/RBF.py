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
#    1、this file's function is to train the lstm model
-------------------------------------------------
"""

from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import numpy as np
from sklearn import model_selection
import tensorflow as tf
import os
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt

# use gpu 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# hyper parameter
NUM = 1000000  # the data length
DROP_OUT1 = 0.1
DROP_OUT2 = 0.1
LR = 0.001  # learning rate
LOAD_NAME = r'../model/BP_5.h5'
SAVE_NAME = r'../model/BP_6.h5'
EPOCHS = 10
BATCH_SIZE = 20
LOAD_MODEL_FLAG = 0  # whether load model
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

    data = data.reshape(-1, 6)
    # data = normalization(data)

    for i in label:
        i = int(i)  # transform into int

    # convert to one-hot code
    y_new = []
    for line in label:
        temp = [0, 0, 0, 0]
        temp[int(line) - 1] = 1
        y_new.append(temp)

    y_new = np.array(y_new)
    y_new = y_new.reshape(-1, 4)
    print(y_new)

    t = pd.DataFrame(y_new)
    t.to_csv('../label_test/y_new.txt', sep='\t', index=False)

    return data, y_new


def train_data(X_train, y_train, X_test, y_test):
    """train the BP model

    Args:
        the data to train and test

    Returns:
        model： the trained BP network
        history: the log of processing

    """
    model = tf.keras.Sequential()
    # rbflayer = RBFLayer(10,
    #                         initializer=InitCentersRandom(X_train),
    #                         betas=2.0,
    #                         input_shape=(6,))
    # model.add(rbflayer)
    # the hidden layer
    # model.add(tf.keras.layers.Dense(10,input_dim=6,activation='sigmoid',kernel_regularizer='L2'))
    # #the output layer
    # model.add(layers.Dense(7,activation='sigmoid',use_bias=True))

    model.add(tf.keras.layers.Dense(10, kernel_regularizer=tf.keras.regularizers.l2(0.01), use_bias=True,
                                    activation='softmax', input_shape=(6,)))
    model.add(tf.keras.layers.Dropout(DROP_OUT1))

    # model.add(tf.keras.layers.Dense(14,kernel_regularizer=tf.keras.regularizers.l2(0.001) ,use_bias=True,activation='relu'))
    # model.add(tf.keras.layers.Dropout(DROP_OUT2))

    model.add(tf.keras.layers.Dense(4, kernel_regularizer=tf.keras.regularizers.l2(0.01), use_bias=True,
                                    activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    if (LOAD_MODEL_FLAG):
        model = tf.keras.models.load_model(LOAD_NAME, custom_objects={'RBFLayer': RBFLayer})

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

    class_weight = {}
    # history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    #              validation_data=(X_test, y_test), validation_freq=1,callbacks=[reduce_lr],shuffle = True,workers = 2,)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

    if SAVE_MODEL_FLAG:
        model.save(filepath=SAVE_NAME)

    return model, history


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
    df = pd.read_csv(r'../data/deleted_total_data2.csv').drop(['x', 'id'], axis=1)
    df = df.reindex(np.random.permutation(df.index))[:NUM]  # random the data

    # data preprocess
    label = np.array(df.loc[:, 'label'])
    data = np.array(df.iloc[:, 0:6])

    data, y_new = data_preprocess(data, label)

    # split the data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data, y_new, shuffle=False, test_size=0.3)

    # train the model
    model, history = train_data(X_train, y_train, X_test, y_test)

    show_pic(history)

    print("----End------")
