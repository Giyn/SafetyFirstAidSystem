"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 9:02:08
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : model_test.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from rbflayer import RBFLayer


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
        temp = [0, 0, 0]
        temp[int(line) - 1] = 1
        label_list.append(temp)

    label_list = np.array(label_list)
    label_list = label_list.reshape(-1, 3)
    result_view = pd.DataFrame(label_list)
    result_view.to_csv('../label_test/predicted_label.txt', sep='\t',
                       index=False)

    return feature, label_list


def evaluate(model, feature, label):
    """
    evaluate the model

    Args:
        model : the traind model
        label : the raw label

    Returns:
        None
    """
    predicted_label = model.predict(feature, batch_size=32)

    label_list = []
    for line in predicted_label:
        index = np.argmax(line)
        temp = [0, 0, 0]
        temp[index] = 1
        label_list.append(temp)
    label_list = np.array(label_list)

    res = np.mean(label_list == label, axis=1)
    print("准确率:", np.sum(res == 1) / len(res))


if __name__ == "__main__":
    print("----Start----")

    df = pd.read_csv('../data/data_generated_by_QG/data_by_QG.csv')

    X = np.array(df.iloc[:, 0:6], dtype=float)
    y = np.array(df.loc[:, 'label'])

    X, y = data_preprocessing(X, y)

    model = tf.keras.models.load_model('../model/RBF_QG.h5',
                                       custom_objects={'RBFLayer': RBFLayer})

    evaluate(model, X, y)

    print("----End------")
