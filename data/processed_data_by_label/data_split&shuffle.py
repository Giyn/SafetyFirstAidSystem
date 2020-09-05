# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_split&shuffle
   Description :
   Author :       Giyn
   date：          2020/9/5 12:12:05
-------------------------------------------------
   Change Activity:
                   2020/9/5 12:12:05
-------------------------------------------------
"""
__author__ = 'Giyn'

import pandas as pd
import numpy as np
import random

total_data_list = []
for i in range(13):
    df = pd.read_csv('processed_data_label_{}.csv'.format(str(i))).drop(['id', 'x'], axis=1)
    data_array = df.values
    data_length = data_array.shape[0]
    left_length = 0
    for each_length in range(128, data_length, 128):
        total_data_list.append(data_array[left_length:each_length])
        left_length = each_length

random.shuffle(total_data_list)

total_data = np.array(total_data_list)
