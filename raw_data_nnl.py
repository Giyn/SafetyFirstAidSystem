# -*- coding:utf-8 -*
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from scipy import  stats
import tensorflow as tf
import  os
def change_list_int(lis):
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
        n_list.append(b)
    return n_list

def change_list_float(lis):
    n_list = []
    print('run')
    for i in range(1, len(lis)):
        if(type(lis[i]).__name__ == 'str'):
            b = lis[i].strip()
            b = float(b)
        elif(type(lis[i]).__name__ == 'float'):
            b = lis[i]
        else:
            b= lis[i]
            b = float(b)
            # print(type(b))
        n_list.append(b)
    return n_list

if __name__ == '__main__':
    path = os.getcwd()
    all_path =path+'/HAPTcsv/all'
    n_time_step = 250
    n_feature = 6
    labels = []
    step = 120
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
    for  i in range(1,len(g2)-n_time_step,step):
        print(type(xs[i]))
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
        reshaped_segments, labels, test_size=0.2, random_state=46)
    print('train and test have splited')
    num_labels = 13
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0025
    N_EPOCHS = 60

    X = tf.placeholder(tf.float32, shape=[None, n_time_step, n_feature], name="input")
    Y = tf.placeholder(tf.float32, shape=[None, num_labels])
    conv1 = tf.layers.conv1d(inputs=X, filters=64, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=5, strides=2, padding='same')
    conv2 = tf.layers.conv1d(inputs=pool1, filters=32, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
    flat = tf.layers.flatten(inputs=conv2)
    logits = tf.layers.dense(inputs=flat, units=13, activation=tf.nn.relu, name="y_")
    L2_LOSS = 0.0015

    l2 = L2_LOSS * \
         sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)) + l2
    #优化器选择adam算法
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    cost_history = np.empty(shape=[1], dtype=float)

    saver = tf.train.Saver()

    history = dict(train_loss=[],
                   train_acc=[],
                   test_loss=[],
                   test_acc=[])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess, "./checkpoint/har.ckpt")
    train_count = len(X_train)
    print('begin to run:')
    #双重循环训练
    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE),
                              range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        _, acc_train, loss_train = sess.run([logits, accuracy, loss], feed_dict={
            X: X_train, Y: y_train})

        _, acc_test, loss_test = sess.run([logits, accuracy, loss], feed_dict={
            X: X_test, Y: y_test})

        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)
        print(f'epoch: {i} test accuracy: {acc_test} loss: {loss_test}')#每个epoch打印一次

    predictions, acc_final, loss_final = sess.run([logits, accuracy, loss], feed_dict={X: X_test, Y: y_test})

    print()
    print(f'final results: accuracy: {acc_final} loss: {loss_final}')
    pickle.dump(predictions, open("predictions2.p", "wb"))#保存文件
    pickle.dump(history, open("history2.p", "wb"))
    sess.close()
