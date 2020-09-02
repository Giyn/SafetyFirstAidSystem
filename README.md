# README

## 文件夹

bpmodel： paddle 下bp神经网络训练得出的结果

HAPT Data Set: 实验中使用的两个数据集之一

RawData： 实验中使用的两个数据集之一

lstmmodel:paddle使用lstm训练得到的模型

checkpoint2： 在tensorflow下使用lstm训练得到的模型

HAPTcsv RDcsv:将RawData数据转换成csv合并

## 代码

### 预处理

analyse.py：分析滑动窗口法下各类型数据多少

preprocessing.py：将txt转化为csv文件并整合acc和gro

merge.py：整合preprocessing.py得到的代码

### tensorflow

lstm_raw.py :使用lstm对raw data训练的代码

nnl.py :使用bp神经网络对HAPT DATA Set训练的代码

raw_data_nnl.py:使用cnn对raw data训练的代码

### paddle

nnl_paddle.py:基于飞桨使用bp神经网络对HAPT DATA SET训练代码

nnl_perdict.py:基于飞桨训练的bp模型进行测试

paddle_raw_data:基于飞桨使用lstm对RawData进行训练代码

## requiment

tensorflow>=1.14

paddle：1.80

USE_CUDA:FALSE

## 模型效果

### tensorflow

| 数据集 | RawData | RawData | HAPTDATASET |
| ------ | ------- | ------- | ----------- |
| 算法   | lstm    | cnn     | BP          |
| 准确率 | 90%     | 87%     | 93%         |

### paddle

| 数据集 | RawData | HAPTDATASET |
| ------ | ------- | ----------- |
| 算法   | lstm    | BP          |
| 准确率 | 80%     | 93%         |