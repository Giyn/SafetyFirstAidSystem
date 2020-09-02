import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#读取数据
X_tr = np.loadtxt('../HAPT Data Set/Train/X_train.txt', delimiter=' ').astype(np.float32)
Y_tr = np.loadtxt('../HAPT Data Set/Train/y_train.txt').astype(np.float32)
Y_tr_svm = Y_tr
Y_tr = np.arange(0, 13) == Y_tr[np.newaxis, :].T
#打乱数据
perm = np.random.permutation(X_tr.shape[0])
X_tr = X_tr[perm, :]
Y_tr = Y_tr[perm, :]
train_size = int(X_tr.shape[0] * 0.8)
X_train = X_tr[:train_size, :]
Y_train = Y_tr[:train_size, :]
X_val = X_tr[train_size:, :]
Y_val = Y_tr[train_size:, :]

X_test = np.loadtxt('../HAPT Data Set/Test/X_test.txt', delimiter=' ').astype(np.float32)
Y_test = np.loadtxt('../HAPT Data Set/Test/y_test.txt').astype(np.float32)
Y_test_svm = Y_test
Y_test = np.arange(0, 13) == Y_test[np.newaxis, :].T

training_epochs = 120
batch_size = 64

X = tf.placeholder(tf.float32, [None, 561])
Y = tf.placeholder(tf.float32, [None, 13])
nhidden = 1024
W1 = tf.Variable(0.001 * np.random.randn(561, nhidden).astype(np.float32), name='weights')
b1 = tf.Variable(0.001 * np.random.randn(nhidden).astype(np.float32), name='bias')

W2 = tf.Variable(0.001 * np.random.randn(nhidden, 13).astype(np.float32), name='weights')
b2 = tf.Variable(0.001 * np.random.randn(13).astype(np.float32), name='bias')
#dropout定义
dropout = tf.placeholder(tf.float32, [])

h0 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(X, W1) + b1), dropout)
#预测结果
pred = tf.matmul(h0, W2) + b2

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.nn.softmax(pred) + 1e-10), reduction_indices=1))
#评价标准
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1)), tf.float32))
#学习率定义
learning_rate = tf.placeholder(tf.float32, [])
#优化器定义
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


loss = np.zeros(training_epochs)
loss_val = np.zeros(training_epochs)
val_acc = np.zeros(training_epochs)
init = tf.global_variables_initializer()

lr = 0.0005
model_path = './model.ckp'
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        #每40个epoch打印一次训练情况
        if epoch % 40 == 0:
            lr /= 2
        num_batches = int(X_train.shape[0] / batch_size)
        curr_loss = 0
        perm = np.random.permutation(X_train.shape[0])
        X_train = X_train[perm, :]
        Y_train = Y_train[perm, :]
        for i in range(num_batches):
            idxs = i * batch_size
            idxe = idxs + batch_size
            batch_x = X_train[idxs:idxe, :]
            batch_y = Y_train[idxs:idxe, :]

            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y, dropout: 0.5, learning_rate: lr})
            curr_loss += c / batch_size
        loss[epoch] = curr_loss
        loss_val[epoch] = cost.eval({X: X_val, Y: Y_val, dropout: 1})
        val_acc[epoch] = accuracy.eval({X: X_val, Y: Y_val, dropout: 1})

        print('Epoch: %d/%d\t cost: %.8f\t val_acc: %.4f' % (epoch + 1, training_epochs, curr_loss, val_acc[epoch]))
    #模型保存
    print("Training Accuracy: ", accuracy.eval({X: X_train, Y: Y_train, dropout: 1}))
    print("Test Accuracy: ", accuracy.eval({X: X_test, Y: Y_test, dropout: 1}))
    saver = tf.train.Saver()
    saver.save(sess, model_path)
    print('save successfuly!')
    plt.plot(loss, 'r-', label='Train Loss')
    plt.plot(loss_val, 'b-', label='Validation Loss')
    plt.tick_params(labelright=True)
    plt.title('Train/Validation Loss vs Epoch')
    plt.ylabel('Train/Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right', shadow=True)
    plt.show()






