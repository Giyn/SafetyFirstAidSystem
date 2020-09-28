# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import  seaborn as sns
from sklearn import  metrics
from sklearn.model_selection import train_test_split
import  os
def tarnsform(inputfile_dir):
    files = os.listdir(inputfile_dir)
    df1 = pd.read_csv(inputfile_dir + '/' + files[0], encoding='utf-8')  # 读取首个csv文件，保存到df1中
    for file in files[1:]:
        df2 = pd.read_csv(inputfile_dir + '/' + file, encoding='utf-8')  # 打开csv文件，注意编码问题，保存到df2中
        df1 = pd.concat([df1, df2], axis=0, ignore_index=True)  # 将df2数据与df1合并
    df1.to_csv(inputfile_dir + '/' + 'total.csv')  # 将结果保存为新的csv文件
path = os.getcwd()

all_path = path+'/HAPTcsv/all'

columns = ['x','label','acc_01','acc_02','acc_03','gry_01','gry_02','gry_03']
df = pd.read_csv(all_path+'/total.csv',header = None,names = columns,encoding='utf-8')
def change_list_int(lis):
    ass = []
    for i in range(1, len(df['x'])):
        if(type(lis[i]).__name__ == 'str'):
            b = lis[i].strip()
            b = int(b)
            print(type(b))
        else:
            b= lis[i]
            # print(type(b))
        ass.append(b)
    return ass

def change_list_float(lis):
    ass = []
    print('run')
    for i in range(1, len(df['x'])):
        if(type(lis[i]).__name__ == 'str'):
            b = lis[i].strip()
            b = float(b)
            print(type(b))
        else:
            b= lis[i]
            # print(type(b))
        ass.append(b)
    return ass

x = df['x'].tolist()
label = df['label'].tolist()
ac1 = df['acc_01'].tolist()
ac2 = df['acc_02'].tolist()
ac3 = df['acc_03'].tolist()
g1 = df['gry_01'].tolist()
g2 = df['gry_02'].tolist()
g3 = df['gry_03'].tolist()
xs = change_list_int(x)
labels = change_list_int(label)
a_1 = change_list_float(ac1)
a_2 = change_list_float(ac2)
a_3 = change_list_float(ac3)
g_1 = change_list_float(g1)
g_2 = change_list_float(g2)
g_3 = change_list_float(g3)
dfs = pd.DataFrame()
dfs['x'] = xs
dfs['label'] =labels
dfs['acc_01'] = a_1
dfs['acc_02'] = a_2
dfs['acc_03'] = a_3
dfs['gry_01'] = g_1
dfs['gry_02'] = g_2
dfs['gry_03'] = g_3
dfs.to_csv(all_path+'/total.csv',encoding='utf-8')#合并