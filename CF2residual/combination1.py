# -*- coding: utf-8 -*-
# =============================================================================
# 重载组合模型
# 作者：张璐
# 时间：2018-03-26
# =============================================================================

from up2down import up2down
from sklearn.preprocessing import scale
from dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mvm import minimal_variance_matching
from clstm1 import LSTModel, InputProducer
import tensorflow as tf


name = 'VTI'
infile = open('rpd/'+name+'.txt', 'r')

center_length = int(infile.readline().strip())
centers = []
for i in range(center_length):
    tmp = infile.readline().strip().split(',')
    centers.append(np.array(tmp, dtype=np.float))


class Config(object):
    input_size = 1
    batch_size = 1
    num_steps = 12
    hidden_size = 32
    output_size = 1
    shift = 5

    num_layers = 1
    keep_prob = 0.8

    learning_rate = 0.001
    training_iter = 12000
    
    
def PPEM(local_segment, centers, Config):
    local_segment = scale(local_segment)
    
    cluster_num = len(centers)
    center_set = centers
    prediction_step_length = Config.shift
    
    # TODO: Match Scheme
    mvm_distance_recorder = np.zeros(cluster_num, dtype=np.float)
    mvm_correspondence_recorder = []
    for i in range(cluster_num):
        mvm_correspondence, mvm_distance = minimal_variance_matching(
                local_segment, center_set[i], skip_elements=2)
        mvm_distance_recorder[i] = mvm_distance
        mvm_correspondence_recorder.append(mvm_correspondence)
            
    mvm_target_index = mvm_distance_recorder.argmin()
    mvm_target = center_set[mvm_target_index]
    mvm_target_cor = mvm_correspondence_recorder[mvm_target_index]
            
    # TODO: prediction
    mvm_length = len(mvm_target)
    path_0, path_1 = list(zip(*mvm_target_cor))
    tmp_length = mvm_length-1-path_1[-1]
            
    if tmp_length >= prediction_step_length:
        res = mvm_target[path_1[-1]+1: path_1[-1]+1+prediction_step_length]
        res = list(map(lambda x: x+(local_segment[-1]-mvm_target[path_1[-1]]), res))
    elif tmp_length >= 1:
        res_1 = mvm_target[path_1[-1]+1:]
        res_2 = [mvm_target[-1]] * (prediction_step_length - tmp_length)
        res = np.append(res_1, res_2)
        res = list(map(lambda x: x+(local_segment[-1]-mvm_target[path_1[-1]]), res))
    else:
        res = [local_segment[-1]] * prediction_step_length
    return res


data_dir = '../data/'
train_present = 0.8


price = pd.read_csv(os.path.join(data_dir, name+'.csv'))
close_price = price.iloc[:, 4].values
close_price = scale(close_price)
close_price_length = len(close_price)

tt_threshold = int(close_price_length * train_present)
train_data = close_price[:tt_threshold]
test_data = close_price[tt_threshold:]

train_producer = InputProducer(train_data, Config)
with tf.variable_scope("Model"):
    T_train_model = LSTModel(is_training=True, config=Config, input_=train_producer)

test_producer = InputProducer(test_data, Config)
with tf.variable_scope("Model", reuse=True):
    test_model = LSTModel(is_training=False, config=Config, input_=test_producer)
    
with tf.variable_scope("Model", reuse=True):
    train_model = LSTModel(is_training=False, config=Config, input_=train_producer)

    

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir="sv_log/")

with sv.managed_session() as sess:
    
    sess.run(init)

    for i in range(Config.training_iter):
        sess.run(T_train_model.optimizer)
    
    sv.saver.save(sess, 'model_save/')
    
    prd_len = 1000

    Train_Right = np.zeros(Config.shift, dtype=np.float)
    Train_RMSE = np.zeros(Config.shift, dtype=np.float)

    for i in range(prd_len):
        x, y, p = sess.run([train_model.input, train_model.valid, train_model.predict])
        x = x.reshape(-1)
        y = y.reshape(-1)
        p = p.reshape(-1)
        
        lax = x[-1]
        for j in range(Config.shift):
            pos = -Config.shift+j
            if ((y[pos]-lax)*(p[pos]-lax) > 0):
                Train_Right[j] += 1
            Train_RMSE[j] += pow(y[pos]-p[pos], 2)
            
        
    #print(x, y, p)

    print("Train_Right:", Train_Right/prd_len)
    print("Train_RMSE:", np.sqrt(Train_RMSE/prd_len))
    
    Test_Right = np.zeros(Config.shift, dtype=np.float)
    Test_RMSE = np.zeros(Config.shift, dtype=np.float)
    PPEM_Right = np.zeros(Config.shift, dtype=np.float)
    PPEM_RMSE = np.zeros(Config.shift, dtype=np.float)
    Comb_Right = np.zeros(Config.shift, dtype=np.float)
    Comb_RMSE = np.zeros(Config.shift, dtype=np.float)
    
    PPEM_index = 0

    for i in range(prd_len):
        x, y, p = sess.run([test_model.input, test_model.valid, test_model.predict])
        x = x.reshape(-1)
        y = y.reshape(-1)
        p = p.reshape(-1)
        
        y = y[-Config.shift:]
        p = p[-Config.shift:]
        ppem = PPEM(x, centers, Config)
        
        
        lax = x[-1]
        for j in range(Config.shift):
            if ((y[j]-lax)*(p[j]-lax) > 0):
                Test_Right[j] += 1
            Test_RMSE[j] += pow(y[j]-p[j], 2)
            
            if (ppem[j]-lax) != 0:
                PPEM_index += 1
                if (ppem[j]-lax)*(y[j]-lax) > 0:
                    PPEM_Right[j] += 1
                PPEM_RMSE[j] += pow(y[j]-ppem[j], 2)
                # 线性加权组合模型
                comb = (p+ppem)/2
            else:
                comb = p
            
            if ((y[j]-lax)*(comb[j]-lax) > 0):
                Comb_Right[j] += 1
            Comb_RMSE[j] += pow(y[j]-p[j], 2)
            
    
    PPEM_index /= 5.0
    print("Test_Right:", Test_Right/prd_len)
    print("Test_RMSE:", np.sqrt(Test_RMSE/prd_len))
    print("PPEM_Right:", PPEM_Right/PPEM_index)
    print("PPEM_RMSE:", np.sqrt(PPEM_RMSE/PPEM_index))
    print("Comb_Right:", Comb_Right/prd_len)
    print("Comb_RMSE:", np.sqrt(Comb_RMSE/prd_len))
        
        











    
            
            
            