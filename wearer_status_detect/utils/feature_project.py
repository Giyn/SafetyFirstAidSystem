"""
-------------------------------------
# -*- coding: utf-8 -*-
# @Time    : 2020/10/25 9:32:41
# @Author  : Giyn
# @Email   : giyn.jy@gmail.com
# @File    : feature_project.py
# @Software: PyCharm
-------------------------------------
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def sliding_window(load_data_path, label, step):
    dataSet_sit = pd.read_csv(load_data_path)

    new_dataSet_list = []

    left_bound = 0
    right_bound = 50
    while right_bound <= 40000:
        each_df = dataSet_sit[left_bound:right_bound]

        each_df_x_1 = np.mean(each_df['feature_1'].values)
        each_df_x_2 = np.mean(each_df['feature_2'].values)
        each_df_x_3 = np.mean(each_df['feature_3'].values)
        each_df_x_4 = np.mean(each_df['feature_4'].values)
        each_df_x_5 = np.mean(each_df['feature_5'].values)
        each_df_x_6 = np.mean(each_df['feature_6'].values)

        each_df_x_7 = np.var(each_df['feature_1'].values)
        each_df_x_8 = np.var(each_df['feature_2'].values)
        each_df_x_9 = np.var(each_df['feature_3'].values)

        new_dataSet_list.append(
            [each_df_x_1, each_df_x_2, each_df_x_3, each_df_x_4,
             each_df_x_5, each_df_x_6, each_df_x_7, each_df_x_8,
             each_df_x_9, label])

        left_bound += step
        right_bound += step

    columns = ['feature_{}'.format(str(i)) for i in range(1, 10)]
    columns.append('label')

    return pd.DataFrame(new_dataSet_list, columns=columns)


if __name__ == '__main__':
    sit_data = sliding_window("../data/data_generated_by_QG/dataSet_sit.csv", 0, 10)
    walk_data = sliding_window("../data/data_generated_by_QG/dataSet_walk.csv", 1, 10)
    run_data = sliding_window("../data/data_generated_by_QG/dataSet_run.csv", 2, 10)
    fall_data = sliding_window("../data/data_generated_by_QG/dataSet_fall.csv", 3, 10)

    dataSet = pd.concat([sit_data, walk_data, run_data, fall_data])
    dataSet = shuffle(dataSet)
    dataSet.to_csv("../data/data_generated_by_QG/dataSet_QG.csv", index=False)
