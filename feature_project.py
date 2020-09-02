# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:26:20 2020

Auther: Giyn
GitHub: https://github.com/Giyn
Email: giyn.jy@gmail.com

"""

import pandas as pd

data_path = './HAPTcsv/all/new_total.csv'

data = pd.read_csv(data_path)[0:1000]

print(data)
