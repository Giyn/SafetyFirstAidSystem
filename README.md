# README

## 文件夹

bpmodel: paddle 下 bp 神经网络训练得出的结果

HAPT Data Set: 实验中使用的两个数据集之一

RawData: 实验中使用的两个数据集之一

lstmmodel: paddle 使用 lstm 训练得到的模型

checkpoint2: 在 tensorflow 下使用 lstm 训练得到的模型

HAPTcsv RDcsv: 将 RawData 数据转换成 csv 合并

## 代码

### 预处理

analyse.py: 分析滑动窗口法下各类型数据多少

preprocessing.py: 将 txt 转化为 csv 文件并整合 acc 和 gro

merge.py: 整合 preprocessing.py 得到的代码

### tensorflow

lstm_raw.py: 使用 lstm 对 raw data 训练的代码

nnl.py: 使用 bp 神经网络对 HAPT DATA Set 训练的代码

raw_data_nnl.py: 使用 cnn 对 raw data 训练的代码

### paddle

nnl_paddle.py: 基于飞桨使用 bp 神经网络对 HAPT DATA SET 训练代码

nnl_perdict.py: 基于飞桨训练的 bp 模型进行测试

paddle_raw_data: 基于飞桨使用 lstm 对 RawData 进行训练代码

## requiment

tensorflow>=1.14

paddle: 1.80

USE_CUDA: FALSE

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