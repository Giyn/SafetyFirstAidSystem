# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2020/9/26 16:49
# @USER     : 86199
# @File         : csv_to_txt
# @Software : PyCharm
# @license  : Copyright(C), xxxCompany
# @Author   : 张平路
------------------------------------------------- 
# @Attantion：
#    1、this file's function is to transform data from csv into txt,in order to transplant code from pycharm to Raspberry Pi
#    2、
#    3、
-------------------------------------------------
"""
import numpy as np
import pandas as pd

if __name__ == "__main__":
    print("----Start----")

    f = pd.read_csv(r'../data/new_total.csv').drop(['x', 'id'], axis=1)[:10240]

    f.to_csv('test1.txt', sep='\t', index=False)

    file = open("test1.txt")
    data = []
    f = file.readlines()
    n = 0
    for i in f:
        if n == 0:  # rid off first line
            n = n + 1
            continue

        j = i.split('\t')
        j[6] = j[6][0:-1]  # rid off the string '\n'
        data.append(j)
    print(data)

    print("----End------")
