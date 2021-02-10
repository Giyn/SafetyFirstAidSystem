# 源码说明

本文件对基于六轴传感器数据的神经网络训练代码进行解释说明。



## data 文件夹

all_data 是原始的 txt 文件转换为 csv 文件，工作由 18 级师兄所做。



data_split_by_label 是将数据按照 label 进行切分。



processed_data_by_label 是对上一个文件夹的数据进行扩列，增加了 74 列数据，但在后续训练中发现是噪声，已经弃用。



raw_data 存放的是网络下载来的原始数据。

[原始数据集网站](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)



total_data.csv 文件是将所有数据汇合在一起。



deleted_total_data.csv 将除了 1,2,3,4,5,6,0 的标签全部删除，并将标签为 0 删除一半，解决样本不均衡的问题。



deleted_total_data2.csv 将标签 2,3,0 全部删除，并重新排序，最后结果为 1 为 walking，2 为 sitting，3 为 laying，4 为 running，**是最终的训练集**。



deleted_total_data3.csv 记不清进行了什么操作，已经废弃。



## data_processing 文件夹

csv_to_txt.py 文件夹是用于向树莓派移植的测试代码，因为树莓派中没有 csv 格式。



data_split_by_label.py 文件用于生成 data_split_by_label 文件夹里的数据。



merge.py 用于合并 csv 表格。



preprocessing.py 文件用于数据预处理。



rbflayer 是自定义层，用于 csv_to_txt.py 的配置



## feature_project

用于扩充数据维度，因为效果不好被废弃。



## label_test

存储生成的测试结果。



## model

用于存放模型，最终模型为 BP_5.h5**（虽然用 BP 命名，但是 RBF 神经网络）**



## model_training

BP_nn.py 用于训练 BP 网络，效果不好。

BP_test.py 测试生成的 BP 网络。

**注意！！！所有的 test 文件是因为我当时的代码失误而写，测试结果训练最后的网络评估是正确的，不必看 test，剩余 test 的文件不再说明！！！**



lstm.py 用于训练 lstm，训练数据 128 个为一组投放，训练标签为后三十二个标签的众数。效果较好，准确率能达到 95%，但由于代码失误废弃，实际也可以投入实际使用。

RBF.py 用于训练 RBF 网络，收敛速度快，准确率高，成为最终选用的模型。
