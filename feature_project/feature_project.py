"""
@date:created on 2020/8/31 16:14
@author:zpl
"""

import os
import numpy as np
import pandas as pd
import csv

for num in range(13):
    path = 'D:\pycharm code/SmartSafetyHelmet/split_data/data_spilt_by_label/df_label_'+str(num)+'.csv'
    df_total_length = pd.read_csv(path, encoding='utf-8').shape[0]
    flag = True
    n = 0
    while flag:
        begin = 0
        end = 1024

        for j in range(1024):
            # open the data

            fr = pd.read_csv(path, encoding='utf-8')[13072 * n + 128 * j:13072 * n + 128 * (j + 1)]

            # the shape 1 of data
            df_length = fr.shape[0]
            if 131072 * n + 128 * (j + 1) > df_total_length:
                flag = False
                break

            # create a new dataframe to store new feature
            add_dataframe = pd.DataFrame(data=np.zeros((df_length, 74)),
                                         columns=['acc_sum', 'gry_sum', 'acc_x_sd', 'acc_y_sd', 'acc_z_sd', 'gry_x_sd',
                                                  'gry_y_sd'
                                             , 'gry_z_sd', 'acc_x_range', 'acc_y_range', 'acc_z_range', 'gry_x_range',
                                                  'gry_y_range'
                                             , 'gry_z_range', 'acc_x_per25', 'acc_x_per50', 'acc_x_per75',
                                                  'acc_x_persub',
                                                  'acc_y_per25', 'acc_y_per50', 'acc_y_per75', 'acc_y_persub',
                                                  'acc_z_per25', 'acc_z_per50', 'acc_z_per75', 'acc_z_persub',
                                                  'gry_x_per25', 'gry_x_per50', 'gry_x_per75', 'gry_x_persub',
                                                  'gry_y_per25', 'gry_y_per50', 'gry_y_per75', 'gry_y_persub',
                                                  'gry_z_per25', 'gry_z_per50', 'gry_z_per75', 'gry_z_persub',
                                                  'acc_x_cor', 'acc_y_cor', 'acc_z_cor', 'gry_x_cor', 'gry_y_cor',
                                                  'gry_z_cor', 'acc_x_skew', 'acc_y_skew', 'acc_z_skew', 'gry_x_skew',
                                                  'gry_y_skew',
                                                  'gry_z_skew', 'acc_x_engy', 'acc_y_engy', 'acc_z_engy', 'gry_x_engy',
                                                  'gry_y_engy',
                                                  'gry_z_engy', 'acc_x_log_engy', 'acc_y_log_engy', 'acc_z_log_engy',
                                                  'gry_x_log_engy', 'gry_y_log_engy',
                                                  'gry_z_log_engy', 'acc_x_poz', 'acc_y_poz', 'acc_z_poz', 'gry_x_poz',
                                                  'gry_y_poz', 'gry_z_poz',
                                                  'acc_xy_cor', 'acc_xz_cor', 'acc_yz_cor', 'gry_xy_cor', 'gry_xz_cor',
                                                  'gry_yz_cor'])

            acc_x_total = np.array(fr.loc[:]['acc_01'])
            # sort the col
            acc_x_total = np.sort(acc_x_total)
            # compute the col range
            acc_x_total_range = acc_x_total[-1] - acc_x_total[0]
            # compute the col avg
            acc_x_total_avg = acc_x_total.sum() / df_length
            # the madian of array
            acc_x_total_median = np.median(acc_x_total)
            # the 25% percentile
            acc_x_25_perc = np.percentile(acc_x_total, 25)
            # the 75% percentile
            acc_x_75_perc = np.percentile(acc_x_total, 75)

            acc_y_total = np.array(fr.loc[:]['acc_02'])
            acc_y_total = np.sort(acc_y_total)
            acc_y_total_range = acc_y_total[-1] - acc_y_total[0]
            acc_y_total_avg = acc_y_total.sum() / df_length
            acc_y_total_median = np.median(acc_y_total)
            acc_y_25_perc = np.percentile(acc_y_total, 25)
            acc_y_75_perc = np.percentile(acc_y_total, 75)

            acc_z_total = np.array(fr.loc[:]['acc_03'])
            acc_z_total = np.sort(acc_z_total)
            acc_z_total_range = acc_z_total[-1] - acc_z_total[0]
            acc_z_total_avg = acc_z_total.sum() / df_length
            acc_z_total_median = np.median(acc_z_total)
            acc_z_25_perc = np.percentile(acc_z_total, 25)
            acc_z_75_perc = np.percentile(acc_z_total, 75)

            gry_x_total = np.array(fr.loc[:]['gry_01'])
            gry_x_total = np.sort(gry_x_total)
            gry_x_total_range = gry_x_total[-1] - gry_x_total[0]
            gry_x_total_avg = gry_x_total.sum() / df_length
            gry_x_total_median = np.median(gry_x_total)
            gry_x_25_perc = np.percentile(gry_x_total, 25)
            gry_x_75_perc = np.percentile(acc_x_total, 75)

            gry_y_total = np.array(fr.loc[:]['gry_02'])
            gry_y_total = np.sort(gry_y_total)
            gry_y_total_range = gry_y_total[-1] - gry_y_total[0]
            gry_y_total_avg = gry_y_total.sum() / df_length
            gry_y_total_median = np.median(gry_y_total)
            gry_y_25_perc = np.percentile(gry_y_total, 25)
            gry_y_75_perc = np.percentile(acc_y_total, 75)

            gry_z_total = np.array(fr.loc[:]['gry_03'])
            gry_z_total = np.sort(gry_z_total)
            gry_z_total_range = gry_z_total[-1] - gry_z_total[0]
            gry_z_total_avg = gry_z_total.sum() / df_length
            gry_z_total_median = np.median(gry_z_total)
            gry_z_25_perc = np.percentile(gry_z_total, 25)
            gry_z_75_perc = np.percentile(acc_z_total, 75)

            # the standard deviation
            acc_x_sd = 0
            acc_y_sd = 0
            acc_z_sd = 0
            gry_x_sd = 0
            gry_y_sd = 0
            gry_z_sd = 0
            # 一阶线性相关性
            tempacc_cor_x1 = 0
            tempacc_cor_x2 = 0
            tempacc_cor_y1 = 0
            tempacc_cor_y2 = 0
            tempacc_cor_z1 = 0
            tempacc_cor_z2 = 0
            tempgry_cor_y2 = 0
            tempgry_cor_z1 = 0
            tempgry_cor_z2 = 0
            tempgry_cor_x1 = 0
            tempgry_cor_x2 = 0
            tempgry_cor_y1 = 0

            # skewness
            tempacc_skew_x1 = 0
            tempacc_skew_x2 = 0
            tempacc_skew_y1 = 0
            tempacc_skew_y2 = 0
            tempacc_skew_z1 = 0
            tempacc_skew_z2 = 0
            tempgry_skew_y2 = 0
            tempgry_skew_z1 = 0
            tempgry_skew_z2 = 0
            tempgry_skew_x1 = 0
            tempgry_skew_x2 = 0
            tempgry_skew_y1 = 0

            # energy
            acc_x_engy = 0
            acc_y_engy = 0
            acc_z_engy = 0
            gry_x_engy = 0
            gry_y_engy = 0
            gry_z_engy = 0

            # l0_energy
            acc_x_lo_engy = 0
            acc_y_lo_engy = 0
            acc_z_lo_engy = 0
            gry_x_lo_engy = 0
            gry_y_lo_engy = 0
            gry_z_lo_engy = 0

            # points over zero
            acc_x_poz = 0
            acc_y_poz = 0
            acc_z_poz = 0
            gry_x_poz = 0
            gry_y_poz = 0
            gry_z_poz = 0

            for i in range(df_length):
                print('第(' + str(13072 * n + 128 * j + i + 1) + '/' + str(df_total_length) + ')次读取')

                # calc the sum of acc vec
                acc_x = fr.iloc[i]['acc_01']
                acc_y = fr.iloc[i]['acc_02']
                acc_z = fr.iloc[i]['acc_03']
                acc_sum = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
                add_dataframe.iloc[i]['acc_sum'] = acc_sum

                acc_x_sd += (acc_x - acc_x_total_avg) ** 2
                acc_y_sd += (acc_y - acc_y_total_avg) ** 2
                acc_z_sd += (acc_z - acc_z_total_avg) ** 2

                # calc the sum of gro vec
                gry_x = fr.iloc[i]['gry_01']
                gry_y = fr.iloc[i]['gry_02']
                gry_z = fr.iloc[i]['gry_03']
                gry_sum = np.sqrt(gry_x ** 2 + gry_y ** 2 + gry_z ** 2)
                add_dataframe.iloc[i]['gry_sum'] = gry_sum

                gry_x_sd += (gry_x - gry_x_total_avg) ** 2
                gry_y_sd += (gry_y - gry_y_total_avg) ** 2
                gry_z_sd += (gry_z - gry_z_total_avg) ** 2

                # first order lag correlaation
                if i != 0:
                    tempacc_cor_x1 += (acc_x - acc_x_total_avg) * (fr.iloc[i - 1]['acc_01'] - acc_x_total_avg)
                    tempacc_cor_y1 += (acc_y - acc_y_total_avg) * (fr.iloc[i - 1]['acc_02'] - acc_y_total_avg)
                    tempacc_cor_z1 += (acc_z - acc_z_total_avg) * (fr.iloc[i - 1]['acc_03'] - acc_z_total_avg)
                    tempgry_cor_x1 += (gry_x - gry_x_total_avg) * (fr.iloc[i - 1]['gry_01'] - gry_x_total_avg)
                    tempgry_cor_y1 += (gry_y - gry_y_total_avg) * (fr.iloc[i - 1]['gry_02'] - gry_y_total_avg)
                    tempgry_cor_z1 += (gry_z - gry_z_total_avg) * (fr.iloc[i - 1]['gry_03'] - gry_z_total_avg)

                tempacc_cor_x2 += (acc_x - acc_x_total_avg) ** 2
                tempacc_cor_y2 += (acc_x - acc_y_total_avg) ** 2
                tempacc_cor_z2 += (acc_x - acc_z_total_avg) ** 2
                tempgry_cor_x2 += (gry_x - gry_x_total_avg) ** 2
                tempgry_cor_y2 += (gry_x - gry_y_total_avg) ** 2
                tempgry_cor_z2 += (gry_x - gry_z_total_avg) ** 2

                tempacc_skew_x1 += (acc_x - acc_x_total_avg) ** 3 / df_length
                tempacc_skew_y1 += (acc_x - acc_y_total_avg) ** 3 / df_length
                tempacc_skew_z1 += (acc_x - acc_z_total_avg) ** 3 / df_length
                tempgry_skew_x1 += (gry_x - gry_x_total_avg) ** 3 / df_length
                tempgry_skew_y1 += (gry_x - gry_y_total_avg) ** 3 / df_length
                tempgry_skew_z1 += (gry_x - gry_z_total_avg) ** 3 / df_length

                acc_x_engy += acc_x ** 2
                acc_y_engy += acc_y ** 2
                acc_z_engy += acc_z ** 2
                gry_x_engy += gry_x ** 2
                gry_y_engy += gry_y ** 2
                gry_z_engy += gry_z ** 2

                acc_x_lo_engy += np.log(acc_x ** 2)
                acc_y_lo_engy += np.log(acc_y ** 2)
                acc_z_lo_engy += np.log(acc_z ** 2)
                gry_x_lo_engy += np.log(gry_x ** 2)
                gry_y_lo_engy += np.log(gry_y ** 2)
                gry_z_lo_engy += np.log(gry_z ** 2)

                if acc_x > acc_x_total_avg:
                    acc_x_poz += 1
                if acc_y > acc_y_total_avg:
                    acc_y_poz += 1
                if acc_z > acc_z_total_avg:
                    acc_z_poz += 1
                if gry_x > gry_x_total_avg:
                    gry_x_poz += 1
                if gry_y > gry_y_total_avg:
                    gry_y_poz += 1
                if gry_z > gry_z_total_avg:
                    gry_z_poz += 1

            tempacc_skew_x2 = (tempacc_cor_x2 / df_length) ** (3 / 2)
            tempacc_skew_y2 = (tempacc_cor_y2 / df_length) ** (3 / 2)
            tempacc_skew_z2 = (tempacc_cor_z2 / df_length) ** (3 / 2)
            tempgry_skew_x2 = (tempgry_cor_x2 / df_length) ** (3 / 2)
            tempgry_skew_y2 = (tempgry_cor_y2 / df_length) ** (3 / 2)
            tempgry_skew_z2 = (tempgry_cor_z2 / df_length) ** (3 / 2)

            add_dataframe.iloc[:]['acc_x_sd'] = acc_x_sd / df_length
            add_dataframe.iloc[:]['acc_y_sd'] = acc_y_sd / df_length
            add_dataframe.iloc[:]['acc_z_sd'] = acc_z_sd / df_length
            add_dataframe.iloc[:]['gry_x_sd'] = gry_x_sd / df_length
            add_dataframe.iloc[:]['gry_y_sd'] = gry_x_sd / df_length
            add_dataframe.iloc[:]['gry_z_sd'] = gry_x_sd / df_length

            add_dataframe.iloc[:]['acc_x_range'] = acc_x_total_range
            add_dataframe.iloc[:]['acc_y_range'] = acc_y_total_range
            add_dataframe.iloc[:]['acc_z_range'] = acc_z_total_range
            add_dataframe.iloc[:]['gry_x_range'] = gry_x_total_range
            add_dataframe.iloc[:]['gry_y_range'] = gry_y_total_range
            add_dataframe.iloc[:]['gry_z_range'] = gry_z_total_range

            # percentile
            add_dataframe.iloc[:]['acc_x_per50'] = acc_x_total_median
            add_dataframe.iloc[:]['acc_x_per25'] = acc_x_25_perc
            add_dataframe.iloc[:]['acc_x_per75'] = acc_x_75_perc
            add_dataframe.iloc[:]['acc_x_persub'] = acc_x_75_perc - acc_x_25_perc

            add_dataframe.iloc[:]['acc_y_per50'] = acc_y_total_median
            add_dataframe.iloc[:]['acc_y_per25'] = acc_y_25_perc
            add_dataframe.loc[:]['acc_y_per75'] = acc_y_75_perc
            add_dataframe.loc[:]['acc_y_persub'] = acc_y_75_perc - acc_y_25_perc

            add_dataframe.loc[:]['acc_z_per50'] = acc_z_total_median
            add_dataframe.loc[:]['acc_z_per25'] = acc_z_25_perc
            add_dataframe.loc[:]['acc_z_per75'] = acc_z_75_perc
            add_dataframe.loc[:]['acc_z_persub'] = acc_z_75_perc - acc_z_25_perc

            add_dataframe.loc[:]['gry_x_per50'] = gry_x_total_median
            add_dataframe.loc[:]['gry_x_per25'] = gry_x_25_perc
            add_dataframe.loc[:]['gry_x_per75'] = gry_x_75_perc
            add_dataframe.loc[:]['gry_x_persub'] = gry_x_75_perc - gry_x_25_perc

            add_dataframe.loc[:]['gry_y_per50'] = gry_y_total_median
            add_dataframe.loc[:]['gry_y_per25'] = gry_y_25_perc
            add_dataframe.loc[:]['gry_y_per75'] = gry_y_75_perc
            add_dataframe.loc[:]['gry_y_persub'] = gry_y_75_perc - gry_y_25_perc

            add_dataframe.loc[:]['gry_z_per50'] = gry_z_total_median
            add_dataframe.loc[:]['gry_z_per25'] = gry_z_25_perc
            add_dataframe.loc[:]['gry_z_per75'] = gry_z_75_perc
            add_dataframe.loc[:]['gry_z_persub'] = gry_z_75_perc - gry_z_25_perc

            add_dataframe.loc[:]['acc_x_cor'] = tempacc_cor_x1 / tempacc_cor_x2
            add_dataframe.loc[:]['acc_y_cor'] = tempacc_cor_y1 / tempacc_cor_y2
            add_dataframe.loc[:]['acc_z_cor'] = tempacc_cor_z1 / tempacc_cor_z2
            add_dataframe.loc[:]['gry_x_cor'] = tempgry_cor_x1 / tempgry_cor_x2
            add_dataframe.loc[:]['gry_y_cor'] = tempgry_cor_y1 / tempgry_cor_y2
            add_dataframe.loc[:]['gry_z_cor'] = tempgry_cor_z1 / tempgry_cor_z2

            add_dataframe.loc[:]['acc_x_skew'] = tempacc_skew_x1 / tempacc_skew_x2
            add_dataframe.loc[:]['acc_y_skew'] = tempacc_skew_y1 / tempacc_skew_y2
            add_dataframe.loc[:]['acc_z_skew'] = tempacc_skew_z1 / tempacc_skew_z2
            add_dataframe.loc[:]['gry_x_skew'] = tempgry_skew_x1 / tempgry_skew_x2
            add_dataframe.loc[:]['gry_y_skew'] = tempgry_skew_y1 / tempgry_skew_y2
            add_dataframe.loc[:]['gry_z_skew'] = tempgry_skew_z1 / tempgry_skew_z2

            add_dataframe.loc[:]['acc_x_engy'] = acc_x_engy
            add_dataframe.loc[:]['acc_y_engy'] = acc_y_engy
            add_dataframe.loc[:]['acc_z_engy'] = acc_z_engy
            add_dataframe.loc[:]['gry_x_engy'] = gry_x_engy
            add_dataframe.loc[:]['gry_y_engy'] = gry_y_engy
            add_dataframe.loc[:]['gry_z_engy'] = gry_z_engy

            add_dataframe.loc[:]['acc_x_log_engy'] = acc_x_lo_engy
            add_dataframe.loc[:]['acc_y_log_engy'] = acc_y_lo_engy
            add_dataframe.loc[:]['acc_z_log_engy'] = acc_z_lo_engy
            add_dataframe.loc[:]['gry_x_log_engy'] = gry_x_lo_engy
            add_dataframe.loc[:]['gry_y_log_engy'] = gry_y_lo_engy
            add_dataframe.loc[:]['gry_z_log_engy'] = gry_z_lo_engy

            add_dataframe.loc[:]['acc_x_poz'] = float(acc_x_poz)
            add_dataframe.loc[:]['acc_y_poz'] = float(acc_y_poz)
            add_dataframe.loc[:]['acc_z_poz'] = float(acc_z_poz)
            add_dataframe.loc[:]['gry_x_poz'] = float(gry_x_poz)
            add_dataframe.loc[:]['gry_y_poz'] = float(gry_y_poz)
            add_dataframe.loc[:]['gry_z_poz'] = float(gry_z_poz)

            acc_x_total_pd = pd.Series(acc_x_total)
            acc_y_total_pd = pd.Series(acc_y_total)
            acc_z_total_pd = pd.Series(acc_z_total)
            gry_x_total_pd = pd.Series(gry_x_total)
            gry_y_total_pd = pd.Series(gry_y_total)
            gry_z_total_pd = pd.Series(gry_z_total)

            add_dataframe.loc[:]['acc_xz_cor'] = acc_x_total_pd.corr(acc_z_total_pd)
            add_dataframe.loc[:]['acc_yz_cor'] = acc_y_total_pd.corr(acc_z_total_pd)
            add_dataframe.loc[:]['acc_xy_cor'] = acc_x_total_pd.corr(acc_y_total_pd)
            add_dataframe.loc[:]['gry_xz_cor'] = gry_x_total_pd.corr(gry_z_total_pd)
            add_dataframe.loc[:]['gry_yz_cor'] = gry_y_total_pd.corr(gry_z_total_pd)
            add_dataframe.loc[:]['gry_xy_cor'] = gry_x_total_pd.corr(gry_y_total_pd)

            # res = pd.concat([fr,add_dataframe],axis=1)\
            path2 = 'D:\pycharm code\SmartSafetyHelmet/add_data/add_data' + str(n + 1) + '_'+str(num)+'.csv'
            flag_csv = os.path.isfile(path2)

            with open(path2, 'a', newline='') as f:
                writer = csv.writer(f)
                if not flag_csv:
                    writer.writerow(
                        ['id', 'x', 'label', 'acc_01', 'acc_02', 'acc_03', 'gry_01', 'gry_02', 'gry_03', 'acc_sum',
                         'gry_sum',
                         'acc_x_sd', 'acc_y_sd', 'acc_z_sd', 'gry_x_sd', 'gry_y_sd'
                            , 'gry_z_sd', 'acc_x_range', 'acc_y_range', 'acc_z_range', 'gry_x_range', 'gry_y_range'
                            , 'gry_z_range', 'acc_x_per25', 'acc_x_per50', 'acc_x_per75', 'acc_x_persub',
                         'acc_y_per25', 'acc_y_per50', 'acc_y_per75', 'acc_y_persub',
                         'acc_z_per25', 'acc_z_per50', 'acc_z_per75', 'acc_z_persub',
                         'gry_x_per25', 'gry_x_per50', 'gry_x_per75', 'gry_x_persub',
                         'gry_y_per25', 'gry_y_per50', 'gry_y_per75', 'gry_y_persub',
                         'gry_z_per25', 'gry_z_per50', 'gry_z_per75', 'gry_z_persub',
                         'acc_x_cor', 'acc_y_cor', 'acc_z_cor', 'gry_x_cor', 'gry_y_cor',
                         'gry_z_cor', 'acc_x_skew', 'acc_y_skew', 'acc_z_skew', 'gry_x_skew', 'gry_y_skew',
                         'gry_z_skew', 'acc_x_engy', 'acc_y_engy', 'acc_z_engy', 'gry_x_engy', 'gry_y_engy',
                         'gry_z_engy', 'acc_x_log_engy', 'acc_y_log_engy', 'acc_z_log_engy', 'gry_x_log_engy',
                         'gry_y_log_engy',
                         'gry_z_log_engy', 'acc_x_poz', 'acc_y_poz', 'acc_z_poz', 'gry_x_poz', 'gry_y_poz', 'gry_z_poz',
                         'acc_xy_cor', 'acc_xz_cor', 'acc_yz_cor', 'gry_xy_cor', 'gry_xz_cor', 'gry_yz_cor'])

                for i in range(df_length):
                    data = list(fr.iloc[i][:]) + list(add_dataframe.iloc[i][:])
                    writer.writerow(data)

        n = n + 1


