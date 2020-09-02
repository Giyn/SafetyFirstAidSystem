# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import  stats
import seaborn as sns
sns.set()
import  os
path = os.getcwd()
def change_list_int(lis):
    n_list = []
    for i in range(1, len((lis))):
        if(type(lis[i]).__name__ == 'str'):
            b = lis[i].strip()
            b = int(b)
        elif (type(lis[i]).__name__ == 'int' ):
            b = lis[i]
        else:
            b= lis[i]
            b =int(b)
        n_list.append(b)
    return n_list
all_path =path+'/HAPTcsv/all'
n_time_step = 250
n_feature = 6
labels = []
step = 60
columns = ['x','label','acc_01','acc_02','acc_03','gry_01','gry_02','gry_03']
df = pd.read_csv(all_path+'/total.csv',header = None,names = columns,encoding='utf-8')
df = df.dropna()
x = df['x'].tolist()
label = df['label'].tolist()
ac1 = df['acc_01'].tolist()
ac2 = df['acc_02'].tolist()
ac3 = df['acc_03'].tolist()
g1 = df['gry_01'].tolist()
g2 = df['gry_02'].tolist()
g3 = df['gry_03'].tolist()
xs = change_list_int(x)
labes = change_list_int(label)

for  i in range(1,len(labes)-n_time_step,step):
    if xs[i+n_time_step] > xs[i]:
        label =stats.mode(labes[i:i + n_time_step])[0][0]
        labels.append(label)
labels = np.array(labels)
df = pd.DataFrame()
df['label'] = labels
ax=df['label'].value_counts().plot(kind='bar')#计算每种类别的个数
plt.title(' class count')
plt.xlabel("activity")
plt.show()