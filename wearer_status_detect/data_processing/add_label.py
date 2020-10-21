"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2020/10/21 19:17:25
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : add_label.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def load_data(file_name, label):
    file = open(file_name)
    data = []
    line = file.readline()
    while line:
        feature = eval(line)
        feature.append(label)
        data.append(feature)
        line = file.readline()

    columns = ['feature_{}'.format(str(i)) for i in range(1, 7)]
    columns.append('label')
    dataSet = pd.DataFrame(np.array(data), columns=columns)

    return dataSet


if __name__ == '__main__':
    dataSet_sit = load_data("../data/data_generated_by_QG/lable0坐.txt", 0)
    dataSet_lie = load_data("../data/data_generated_by_QG/lable1躺.txt", 1)
    dataSet_walk = load_data("../data/data_generated_by_QG/lable2走.txt", 2)
    dataSet_run = load_data("../data/data_generated_by_QG/lable3跑.txt", 3)
    dataSet_sit['label'] = dataSet_sit['label'].map(lambda x: int(x))
    dataSet_lie['label'] = dataSet_lie['label'].map(lambda x: int(x))
    dataSet_walk['label'] = dataSet_walk['label'].map(lambda x: int(x))
    dataSet_run['label'] = dataSet_run['label'].map(lambda x: int(x))

    dataSet = pd.concat([dataSet_sit, dataSet_lie, dataSet_walk, dataSet_run])
    dataSet = shuffle(dataSet)
    dataSet.to_csv("../data/data_generated_by_QG/data_by_QG.csv", index=False)
