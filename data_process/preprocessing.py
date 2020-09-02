"""
This file is used as data preprocessing steps before input pipeline
"""

import os
import glob
import numpy as np
import pandas as pd
import shutil
import re
from scipy.stats import zscore
from absl import logging

logging.set_verbosity(logging.INFO)


def to_csv(path):
    # 将txt文件转化为csv文件
    txt_path = os.path.join(path, 'RawData/*.txt')
    txt_files = glob.glob(txt_path)
    for filename in txt_files:
        txt = np.loadtxt(filename)
        txtDF = pd.DataFrame(txt)
        name = os.path.splitext(filename)
        csv_file = name[0] + '.csv'
        txtDF.to_csv(csv_file, index=False)
        csv_new_file = csv_file.replace('RawData', 'RDcsv')
        shutil.move(csv_file, csv_new_file)
        logging.info('{} is created'.format(csv_new_file))


def label_transform(path):
    #合并标签
    csv_path = path + '/RDcsv/*.csv'
    label_path = path + '/RDcsv/labels.csv'
    csv_files = glob.glob(csv_path)

    for filename in csv_files:
        exp = os.path.split(filename)[1]
        exp = exp.split('.')[0]
        sensor = exp.split('_')[0]

        if sensor == 'acc':
            exp_num = re.findall("\d+", exp)[0]
            user_num = re.findall("\d+", exp)[1]

            acc_file = filename
            gyro_file = acc_file.replace('acc', 'gyro')

            df_acc = pd.read_csv(acc_file)
            df_gyro = pd.read_csv(gyro_file)

            df_labels = pd.read_csv(label_path)
            label_filename = path + '/RDcsv/label_exp' + exp_num + '.csv'
            df_exp = df_labels[df_labels['0'] == int(exp_num)]
            raw = len(df_exp)
            last = df_exp.iloc[raw - 1, 4]
            df = pd.DataFrame(columns=['x', 'label'])
            df['x'] = df['x'].astype(int)
            df['label'] = df['label'].astype(int)
            for i in range(int(last) + 1):
                label = 0
                for index in range(raw):
                    if i >= df_exp.iloc[index, 3] and i <= df_exp.iloc[index, 4]:
                        label = int(df_exp.iloc[index, 2])
                df = df.append({'x': i, 'label': label}, ignore_index=True)

            df.to_csv(label_filename, encoding="utf-8", index=False)

            logging.info('experiment {}, user {}'.format(exp_num, user_num))



def input_merge(path):
    # 合并两类传感器数据
    csv_path = path + '/RDcsv/*.csv'
    csv_files = glob.glob(csv_path)

    for filename in csv_files:
        exp = os.path.split(filename)[1]
        exp = exp.split('.')[0]
        sensor = exp.split('_')[0]

        if sensor == 'acc':
            exp_num = re.findall("\d+", exp)[0]
            user_num = re.findall("\d+", exp)[1]

            acc_file = filename
            gyro_file = acc_file.replace('acc', 'gyro')
            label_file = path + '/RDcsv/label_exp' + exp_num + '.csv'

            df_acc = pd.read_csv(acc_file)
            df_gyro = pd.read_csv(gyro_file)
            df_label = pd.read_csv(label_file)

            df_acc = df_acc.apply(zscore)
            df_gyro = df_gyro.apply(zscore)

            df_acc = df_acc.rename(columns={'0': 'acc_01', '1': 'acc_02', '2': 'acc_03'})
            df_gyro = df_gyro.rename(columns={'0': 'gyro_01', '1': 'gyro_02', '2': 'gyro_03'})
            df = pd.concat([df_label, df_acc, df_gyro], axis=1)

            file_name = path + '/HAPTcsv/HAPT_exp' + exp_num + '_user' + user_num + '.csv'
            df.to_csv(file_name, encoding="utf-8", index=False)
            logging.info('The input_label file {} for experiment {} is created'.format(file_name, exp_num))

if __name__ == '__main__':
    path = os.getcwd()
    to_csv(path)
    label_transform(path)
    input_merge(path)
