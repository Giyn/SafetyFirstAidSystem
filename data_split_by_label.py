# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_split_by_label
   Description :
   Author :       Giyn
   date：          2020/9/2 13:14:55
-------------------------------------------------
   Change Activity:
                   2020/9/2 13:14:55
-------------------------------------------------
"""
__author__ = 'Giyn'

import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')  # log information settings

df = pd.read_csv('./HAPTcsv/all/new_total.csv')
df = df.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8]]

print(df['label'].unique())
df_label_4831 = df[df['label'] == 4831]
print(df_label_4831)

# df_label_0 = df[df['label'] == 0]
# df_label_1 = df[df['label'] == 1]
# df_label_2 = df[df['label'] == 2]
# df_label_3 = df[df['label'] == 3]
# df_label_4 = df[df['label'] == 4]
# df_label_5 = df[df['label'] == 5]
# df_label_6 = df[df['label'] == 6]
# df_label_7 = df[df['label'] == 7]
# df_label_8 = df[df['label'] == 8]
# df_label_9 = df[df['label'] == 9]
# df_label_10 = df[df['label'] == 10]
# df_label_11 = df[df['label'] == 11]
# df_label_12 = df[df['label'] == 12]
#
# df_label = [df_label_0, df_label_1, df_label_2, df_label_3, df_label_4, df_label_5, df_label_6,
#             df_label_7, df_label_8, df_label_9, df_label_10, df_label_11, df_label_12]
#
# df_names = ['df_label_{}'.format(str(i)) for i in range(13)]
#
# for index, each_df in enumerate(df_label):
#     df_path = './HAPTcsv/data_spilt_by_label/' + df_names[index] + '.csv'
#     each_df.to_csv(df_path)
#     logging.info('successfully save data!')
