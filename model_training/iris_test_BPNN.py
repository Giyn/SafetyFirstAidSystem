"""
-------------------------------------------------
# @Time: 2020/9/26 16:49
# @USER: 86199
# @File: csv_to_txt
# @Software: PyCharm
# @license: Copyright(C), xxxCompany
# @Author: 张平路
-------------------------------------------------
# @Attantion：
#    1、this file's function is to test data structure to find the problem where it is.
#    2、if the network work well,there may be some wrong in raw data
#    3、the raw code is copyed from the internet
-------------------------------------------------
"""

from sklearn import datasets
import numpy as np

np.random.seed(seed=7)

iris = datasets.load_iris()
X = iris.data  # 150行4列
Y = iris.target  # 150行1列   注意Y的类别标签是0，1，2，需要对Y进行one-hot独热编码

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

Y_Label = LabelBinarizer().fit_transform(Y)
X_train, Y_train, X_test, Y_test = train_test_split(X, Y_Label, test_size=0.3,
                                                    random_state=42)  # random_state=42一定要设置，否则每次训练集测试划分都不一样结果都不一样

##################################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()  # 建立模型

model.add(Dense(4, activation='relu', input_shape=(4,)))

model.add(Dense(6, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

i = 100

history = model.fit(X_train, X_test, validation_data=(Y_train, Y_test),
                    batch_size=10,
                    epochs=i)  # epochs=20表示模型反复训练20次，
train_result = history.history

t = model.predict(Y_train)
resultsss = model.evaluate(Y_train, Y_test)
print(X)

print(model.predict([[5.1, 3.8, 1.5, 0.3]]))
