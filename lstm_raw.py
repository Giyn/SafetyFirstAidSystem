# -*- coding:utf-8 -*
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy import  stats
import os


def change_list_int(lis):
    #将csv中的数据转化为int类型
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
            # print(type(b))
        n_list.append(b)
    return n_list

def change_list_float(lis):
    #将csv中的数据转换为float类型
    n_list = []
    for i in range(1, len(lis)):
        if(type(lis[i]).__name__ == 'str'):
            b = lis[i].strip()
            b = float(b)
        elif(type(lis[i]).__name__ == 'float'):
            b = lis[i]
        else:
            b= lis[i]
            b = float(b)
        n_list.append(b)
    return n_list

def normalization(data):
    #正则化代码
    avg = np.mean(data)
    max_ = np.max(data)
    min_ = np.min(data)
    result_data = (data - avg) / (max_ - min_)
    return result_data

if __name__ == '__main__':
    path = os.getcwd()
    all_path =path+'/HAPTcsv/all'
    n_time_step = 250
    n_feature = 6
    labels = []
    step = 60
    segments = []
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
    a_1 = change_list_float(ac1)
    a_2 = change_list_float(ac2)
    a_3 = change_list_float(ac3)
    g_1 = change_list_float(g1)
    g_2 = change_list_float(g2)
    g_3 = change_list_float(g3)
    for  i in range(1,len(g_1)-n_time_step,step):
        if xs[i+n_time_step] > xs[i]:
            acx = a_1[i:i+n_time_step]
            acy = a_2[i:i+n_time_step]
            acz = a_3[i:i + n_time_step]
            grx = g1[i:i + n_time_step]
            gry = g2[i:i + n_time_step]
            grz = g3[i:i + n_time_step]
            label =stats.mode(labes[i:i + n_time_step])[0][0]
            segments.append([acx,acy,acz,grx,gry,grz])
            labels.append(label)
    reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_step, n_feature)
    labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        reshaped_segments, labels, test_size=0.25)
    print('train and test have splited')
    N_CLASSES = 13
    N_HIDDEN_UNITS = 32


    def create_LSTM_model(inputs):
        W = {
            'hidden': tf.Variable(tf.random_normal([n_feature, N_HIDDEN_UNITS])),
            'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
        }
        biases = {
            'hidden': tf.Variable(tf.random_normal([N_HIDDEN_UNITS], mean=1.0)),
            'output': tf.Variable(tf.random_normal([N_CLASSES]))
        }

        X = tf.transpose(inputs, [1, 0, 2])
        X = tf.reshape(X, [-1, n_feature])
        hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + biases['hidden'])
        hidden = tf.split(hidden, n_time_step, 0)

        # Stack 2 LSTM layers
        lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)]
        lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

        outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

        # Get output for the last time step
        lstm_last_output = outputs[-1]

        return tf.matmul(lstm_last_output, W['output']) + biases['output']


    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, n_time_step, n_feature], name="input")
    Y = tf.placeholder(tf.float32, [None, N_CLASSES])

    pred_Y = create_LSTM_model(X)

    pred_softmax = tf.nn.softmax(pred_Y, name="y_")

    L2_LOSS = 0.0015
    l2 = L2_LOSS * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels=Y)) + l2
    #学习率选择0.0025
    LEARNING_RATE = 0.0025
    #优化器选用adam算法
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    N_EPOCHS = 50
    BATCH_SIZE = 64

    saver = tf.train.Saver()

    history = dict(train_loss=[],
                   train_acc=[],
                   test_loss=[],
                   test_acc=[])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    train_count = len(X_train)

    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE),
                              range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={
            X: X_train, Y: y_train})

        _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={
            X: X_test, Y: y_test})

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)


        print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')

    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: X_test, Y: y_test})

    print()
    print(f'final results: accuracy: {acc_final} loss: {loss_final}')

    pickle.dump(predictions, open("./checkpoint2/predictions90.p", "wb"))
    pickle.dump(history, open("./checkpoint2/history90.p", "wb"))
    tf.train.write_graph(sess.graph_def, '.', './checkpoint2/har.pbtxt')
    saver.save(sess, save_path="./checkpoint2/har.ckpt")#保存模型
    sess.close()
