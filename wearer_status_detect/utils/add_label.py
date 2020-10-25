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
    dataSet_sit1 = load_data("../data/data_generated_by_QG/静止站立0/data1.txt", 0)
    dataSet_sit2 = load_data("../data/data_generated_by_QG/静止站立0/data2.txt", 0)
    dataSet_sit3 = load_data("../data/data_generated_by_QG/静止站立0/data3.txt", 0)
    dataSet_sit4 = load_data("../data/data_generated_by_QG/静止站立0/data4.txt", 0)

    dataSet_walk1 = load_data("../data/data_generated_by_QG/缓慢行走1/data1.txt", 1)
    dataSet_walk2 = load_data("../data/data_generated_by_QG/缓慢行走1/data2.txt", 1)
    dataSet_walk3 = load_data("../data/data_generated_by_QG/缓慢行走1/data3.txt", 1)
    dataSet_walk4 = load_data("../data/data_generated_by_QG/缓慢行走1/data4.txt", 1)

    dataSet_run1 = load_data("../data/data_generated_by_QG/跑步2/data1.txt", 2)
    dataSet_run2 = load_data("../data/data_generated_by_QG/跑步2/data2.txt", 2)
    dataSet_run3 = load_data("../data/data_generated_by_QG/跑步2/data3.txt", 2)
    dataSet_run4 = load_data("../data/data_generated_by_QG/跑步2/data4.txt", 2)

    dataSet_fall1 = load_data("../data/data_generated_by_QG/躺下摔倒3/data1.txt", 3)
    dataSet_fall2 = load_data("../data/data_generated_by_QG/躺下摔倒3/data2.txt", 3)
    dataSet_fall3 = load_data("../data/data_generated_by_QG/躺下摔倒3/data3.txt", 3)
    dataSet_fall4 = load_data("../data/data_generated_by_QG/躺下摔倒3/data4.txt", 3)

    dataSet_sit1['label'] = dataSet_sit1['label'].map(lambda x: int(x))
    dataSet_sit2['label'] = dataSet_sit2['label'].map(lambda x: int(x))
    dataSet_sit3['label'] = dataSet_sit3['label'].map(lambda x: int(x))
    dataSet_sit4['label'] = dataSet_sit4['label'].map(lambda x: int(x))

    dataSet_walk1['label'] = dataSet_walk1['label'].map(lambda x: int(x))
    dataSet_walk2['label'] = dataSet_walk2['label'].map(lambda x: int(x))
    dataSet_walk3['label'] = dataSet_walk3['label'].map(lambda x: int(x))
    dataSet_walk4['label'] = dataSet_walk4['label'].map(lambda x: int(x))

    dataSet_run1['label'] = dataSet_run1['label'].map(lambda x: int(x))
    dataSet_run2['label'] = dataSet_run2['label'].map(lambda x: int(x))
    dataSet_run3['label'] = dataSet_run3['label'].map(lambda x: int(x))
    dataSet_run4['label'] = dataSet_run4['label'].map(lambda x: int(x))

    dataSet_fall1['label'] = dataSet_fall1['label'].map(lambda x: int(x))
    dataSet_fall2['label'] = dataSet_fall2['label'].map(lambda x: int(x))
    dataSet_fall3['label'] = dataSet_fall3['label'].map(lambda x: int(x))
    dataSet_fall4['label'] = dataSet_fall4['label'].map(lambda x: int(x))

    # data_list = [dataSet_sit1, dataSet_sit2, dataSet_sit3, dataSet_sit4,
    #              dataSet_walk1, dataSet_walk2, dataSet_walk3, dataSet_walk4,
    #              dataSet_run1, dataSet_run2, dataSet_run3, dataSet_run4,
    #              dataSet_fall1, dataSet_fall2, dataSet_fall3, dataSet_fall4]

    dataSet_sit = pd.concat([dataSet_sit1, dataSet_sit2, dataSet_sit3, dataSet_sit4])
    dataSet_walk = pd.concat([dataSet_walk1, dataSet_walk2, dataSet_walk3, dataSet_walk4])
    dataSet_run = pd.concat([dataSet_run1, dataSet_run2, dataSet_run3, dataSet_run4])
    dataSet_fall = pd.concat([dataSet_fall1, dataSet_fall2, dataSet_fall3, dataSet_fall4])

    # dataSet = pd.concat(data_list)
    # dataSet = shuffle(dataSet)

    dataSet_sit.to_csv("../data/data_generated_by_QG/dataSet_sit.csv", index=False)
    dataSet_walk.to_csv("../data/data_generated_by_QG/dataSet_walk.csv", index=False)
    dataSet_run.to_csv("../data/data_generated_by_QG/dataSet_run.csv", index=False)
    dataSet_fall.to_csv("../data/data_generated_by_QG/dataSet_fall.csv", index=False)
