# feature_info

本文件用于说明特征信息

## 背景

此文件夹的数据集是通过 RawData 的原始数据扩增而来，新增特征 74 维。
csv 文件按照标签分类，每 128 个数据为一组进行计算新增特征



## 新增特征信息

**注意没有计算峰密度这一变量**



## 特征名

### 1.和向量

sum

如：acc_x_sum

### 2.标准差

sd

acc_x_sd

### 3.极差

range

如：acc_x_range

### 4.分位点和分位点差

**三个分位点**：

per25

如：acc_x_per25

per50

如：acc_x_per50

per75

如：acc_x_per75

**分位差**

persub

如：acc_x_persub

### 5.一阶滞后相关性

cor

如：acc_x_cor

### 6.偏度

skew

如：acc_x_skew

### 7.能量

engy

如：acc_x_engy

### 8.对数能量

log_engy

如：acc_x_log_engy

### 9.过零点数

poz

如：acc_x_poz

### 10.相关性：

cor(不要和一阶滞后相关性混淆)

如：acc_xy_cor

