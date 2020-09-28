"""
-------------------------------------------------
# @Time: 2020/9/27 20:44
# @USER: 86199
# @File: testbp
# @Software: PyCharm
# @license: Copyright(C), xxxCompany
# @Author: 张平路
-------------------------------------------------
"""
import numpy as np
import pandas as pd
from scipy import stats
import tensorflow as tf
from rbflayer import RBFLayer, InitCentersRandom


# import tensorflow as tf

def normalization(data):
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
    data = normalization(data)
    y_new = []
    for line in label:
        temp = [0, 0, 0, 0, 0, 0, 0]

        temp[int(line)] = 1
        y_new.append(temp)
    y_new = np.array(y_new)
    y_new = y_new.reshape(-1, 7)

    return data, y_new


def evaluate(model, y_new):
    """
    evaluate the model

    Args:
        model : the traind model
        y_new : the raw label

    Returns:
        None
    """
    y_p = model.predict(data, batch_size=32)

    y_p_new = []
    for line in y_p:
        index = np.argmax(line)
        temp = [0, 0, 0, 0, 0, 0, 0]

        temp[index] = 1
        y_p_new.append(temp)
    y_p_new = np.array(y_p_new)
    print(y_p_new)
    t = pd.DataFrame(y_p_new)
    t.to_csv('../label_test/y_p_new.txt', sep='\t', index=False)

    print(y_new)

    res = np.mean(y_p_new == y_new, axis=1)
    print(np.sum(res == 1) / len(res))


if __name__ == "__main__":
    print("----Start----")

    df = pd.read_csv(r'../data/deleted_total_data.csv').drop(['x', 'id'], axis=1)
    df = df.reindex(np.random.permutation(df.index))[:1000]  # random the data

    # data preprocess
    label = np.array(df.loc[:, 'label'])
    data = np.array(df.iloc[:, 1:], dtype=float)

    data, y_new = data_preprocess(data, label)

    model = tf.keras.Sequential()
    model = tf.keras.models.load_model(r'../trained_model/BP_4.h5', custom_objects={'RBFLayer': RBFLayer})
    # np.set_printoptions(precision=4)

    evaluate(model, y_new)

    print("----End------")
