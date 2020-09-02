# -*- coding:utf-8 -*
import pandas as pd
import numpy as np
import  paddle
import  paddle.fluid as fluid
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import  stats
import  os
import tensorflow.compat.v1 as tf
def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost)
    plt.plot(iters, accs,color='green',label=lable_acc)
    plt.legend()
    plt.grid()
    plt.show()
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
            # print(type(b))
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

path = os.getcwd()
all_path =path+'/HAPTcsv/all'
n_time_step = 256
n_feature = 6
labels = []
step = 128
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
for i in range(1, len(g2) - n_time_step, step):
    if xs[i + n_time_step] > xs[i]:
        acx = a_1[i:i + n_time_step]
        acy = a_2[i:i + n_time_step]
        acz = a_3[i:i + n_time_step]
        grx = g1[i:i + n_time_step]
        gry = g2[i:i + n_time_step]
        grz = g3[i:i + n_time_step]
        label = stats.mode(labes[i:i + n_time_step])[0][0]
        segments.append([acx, acy, acz, grx, gry, grz])
        labels.append(label)
reshaped_segments = np.asarray(segments, dtype=np.float32).reshape(-1, n_time_step, n_feature)
X_train, X_test, y_train, y_test = train_test_split(
    reshaped_segments, labels, test_size=0.2)
print('train and test have splited')
BUF_SIZE=128
BATCH_SIZE=64
def reader_createor(data, label):
    def reader():
        for i in  range(len(data)):
            yield data[i,:], label[i]
    return reader

#用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(reader=reader_createor(X_train, y_train),buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
#用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
test_reader = paddle.batch(
    paddle.reader.shuffle(reader=reader_createor(X_test, y_test), buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)
train_data=reader_createor(X_train, y_train)

def lstm_net(data,label, input_dim=6, class_dim=13,  hid_dim=128,hid_dim2 = 64):
    fc0 = fluid.layers.fc(input=data, size=hid_dim*4 )
    lstm_h, c = fluid.layers.dynamic_lstm(input=fc0, size=hid_dim * 4, is_reverse=False,gate_activation = 'tanh')
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.leaky_relu(lstm_max)
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='leaky_relu')
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax')
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction

def train(train_reader,network,use_cuda,save_dirname,lr=0.2,pass_num=50):
    all_train_iter = 0
    all_train_iters = []
    all_train_costs = []
    all_train_accs = []
    # 输入层
    data = fluid.layers.data(
        name="x", shape=[None,6], dtype="float32", lod_level=1)

    # 标签层
    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    # 网络结构
    cost, acc, prediction = network(data, label)

    # 优化器
    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    # 设备、执行器、feeder 定义
    test_program = fluid.default_main_program().clone(for_test=True)
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    # 模型参数初始化
    exe.run(fluid.default_startup_program())

    # 双层循环训练
    # 外层 epoch
    for pass_id in range(pass_num):
        i = 0
        for batch_id, data in enumerate(train_reader()):
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                              feed=feeder.feed(data),
                                              fetch_list=[cost, acc])
            all_train_iter = all_train_iter + BATCH_SIZE
            all_train_iters.append(all_train_iter)
            all_train_costs.append(avg_cost_np[0])
            all_train_accs.append(avg_acc_np[0])
            if batch_id % 50 == 0:
                print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %(pass_id, batch_id, avg_cost_np[0], avg_acc_np[0]))
        epoch_model = save_dirname

        fluid.io.save_inference_model(epoch_model, ["x", "label"], acc, exe)
    print('train end')
    # 进行测试
    test_accs = []
    test_costs = []
    # 每训练一轮 进行一次测试
    for batch_id, data in enumerate(test_reader()):  # 遍历test_reader
        # print(batch_id)
        test_cost, test_acc = exe.run(program=test_program,  # 执行训练程序
                                      feed=feeder.feed(data),  # 喂入数据
                                      fetch_list=[cost, acc])  # fetch 误差、准确率
        test_accs.append(test_acc[0])  # 每个batch的准确率
        test_costs.append(test_cost[0])  # 每个batch的误差

    # 求测试结果的平均值
    test_cost = (sum(test_costs) / len(test_costs))  # 每轮的平均误差
    test_acc = (sum(test_accs) / len(test_accs))  # 每轮的平均准确率
    print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
    draw_train_process("lstm_training", all_train_iters, all_train_costs, all_train_accs, "trainning cost", "trainning acc")
train(train_reader,lstm_net,use_cuda=False,save_dirname="lstm_model",lr=0.005,pass_num=40)
