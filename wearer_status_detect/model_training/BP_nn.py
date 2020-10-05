"""
-------------------------------------------------
# @Time: 2020/9/28 9:50
# @USER: 86199
# @File: rbflayer
# @Software: PyCharm
# @license: Copyright(C), xxxCompany
# @Author: 张平路
-------------------------------------------------
"""

from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
import pandas as pd
import random
import numpy as np
from sklearn import model_selection
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn import preprocessing
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
model = tf.keras.Sequential()

model.add(
    layers.LSTM(52, input_shape=(128, 80), return_sequences=True, use_bias=True,
                dropout=0.1))
# model.add(layers.LSTM(52,input_shape=(256,),use_bias=True,dropout=0.1,return_sequences=True))
model.add(layers.LSTM(26, input_shape=(52,), use_bias=True, dropout=0.1))

model.add(layers.Dense(13, activation='relu', use_bias=True))

model.compile(optimizer=tf.keras.optimizers.Adam(0.000001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
total_data_list = []
for i in range(13):
    print('第{}次读取'.format(i + 1))
    df = pd.read_csv(
        r'../data/processed_data_by_label/processed_data_label_{}.csv'.format(
            str(i))).drop(['id', 'x'],
                          axis=1)

    data_array = df.values
    data_length = data_array.shape[0]
    left_length = 0
    for each_length in range(128, data_length, 128):
        total_data_list.append(data_array[left_length:each_length])
        left_length = each_length

random.shuffle(total_data_list)

total_data = np.array(total_data_list)

data = total_data[:, :, 1:]
label = total_data[:, :, 0]
#
# total_data = np.array(total_data_list).reshape(-1,80)
# total_data = preprocessing.scale(total_data)


label = np.mean(label, axis=1)
label = np.array(label, dtype=int)

# print(fr.shape)
# if i == 0:
#     data = fr
#
# else:
#     data = pd.concat([data, fr], axis=1)
#
# X = np.array(fr.iloc[:, 3:]).reshape(int(fr.shape[0]/128),128,80)
# print(X.shape)
# y = np.array(fr.iloc[:X.shape[0], 2],dtype=int).reshape(-1,1)
y_new = []
for line in label:
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp[line] = 1
    y_new.append(temp)
y_new = np.array(y_new)
# print(y_new)


X_train, X_test, y_train, y_test = model_selection.train_test_split(data, y_new,
                                                                    shuffle=False,
                                                                    test_size=0.3)
# print(X_train.shape)


# model.save('test1.h5')
# model = tf.keras.models.load_model('test1.h5')


reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

model.fit(X_train, y_train, epochs=1, batch_size=128,
          validation_data=(X_test, y_test), validation_freq=1,
          callbacks=[reduce_lr], shuffle=True, workers=2)

# model.evaluate(X_test, y_test, batch_size=32)
y_p = model.predict(X_test, batch_size=32)
y_p_new = []
for line in y_p:
    index = np.argmax(line)
    temp = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    temp[index] = 1
    y_p_new.append(temp)
y_p_new = np.array(y_p_new)
print(y_p_new)
model.evaluate(X_test, y_test, batch_size=32)
# model.save('test1.h5')
tf.saved_model.save(model, "save_test")

# for i in range(5):
#     fr = pd.read_csv('./data/processed_data_by_label/processed_data_label_' + str(i) + '.csv')
#     X = np.array(fr.iloc[:, 3:]).reshape(int(fr.shape[0]/128),128,80)
#     y = np.array(fr.iloc[:, 2]).reshape(-1, 1)
#     print(y)
#     new_model = tf.keras.models.load_model('test.h5')
#     y_p = model.predict(X, batch_size=32)
#     print(y_p)
#     print(np.mean(y == y_p))
