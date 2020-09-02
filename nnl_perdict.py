# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import  paddle.fluid as fluid

use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
model_save_dir = "bpmodel"
x  = np.loadtxt('../HAPT Data Set/Test/X_test.txt', delimiter=' ').astype(np.float32)                        #将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
x = np.array(x).reshape(-1,561)




infer_exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

with fluid.scope_guard(inference_scope):
    #获取训练好的模型
    #从指定目录中加载 推理model(inference model)
    [inference_program,                                            #推理Program
     feed_target_names,                                            #是一个str列表，它包含需要在推理 Program 中提供数据的变量的名称。
     fetch_targets] = fluid.io.load_inference_model(model_save_dir,#fetch_targets：是一个 Variable 列表，从中我们可以得到推断结果。model_save_dir：模型保存的路径
                                                    infer_exe)     #infer_exe: 运行 inference model的 executor


    results = infer_exe.run(program=inference_program,               #运行推测程序
                   feed={feed_target_names[0]: x},           #喂入要预测的x
                   fetch_list=fetch_targets)                   #得到推测结果,
    # 获取概率最大的label
    lab = np.argsort(results)                                  #argsort函数返回的是result数组值从小到大的索引值
    dictActivity = {0:'None' ,1: 'WALKING', 2: 'WALKING_UPSTAIRS',3:'WALKING_DOWNSTAIRS',4:'SITTING',5:'STANDING ',6:'LAYING ',7:'STAND_TO_SIT ',8:'SIT_TO_STAND',9:'SIT_TO_LIE'
                    ,10:"LIE_TO_SIT",11:'STAND_TO_LIE',12:'LIE_TO_STAND'}
    print("该动作的预测结果的label为:" + dictActivity[lab[0][0][-1]])     #预测结果为standing预测正确
