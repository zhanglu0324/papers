# -*- coding: utf-8 -*-
# =============================================================================
# 训练LSTM神经网络
# 作者：张璐
# 时间：2018-03-25
# =============================================================================

from sklearn.preprocessing import scale
from LSTM import Config, LSTModel, InputProducer
import numpy as np
import tensorflow as tf
import os
import pandas as pd


# =============================================================================
# # no summary, no need
# =============================================================================
# os.system("rm -rf sv_log/*")
# os.system("rm -rf model_save/")
# =============================================================================
# =============================================================================

data_dir = '../data/'
name = 'AAPL'
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
    train_model = LSTModel(is_training=True, config=Config, input_=train_producer)

test_producer = InputProducer(test_data, Config)
with tf.variable_scope("Model", reuse=True):
    test_model = LSTModel(is_training=False, config=Config, input_=test_producer)
    

init = tf.global_variables_initializer()

sv = tf.train.Supervisor(logdir="sv_log/")
with sv.managed_session() as sess:
    
# =============================================================================
# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(sess=sess, coord=coord)
# 
# =============================================================================
# =============================================================================
# # 暂时不搞这些了
# summary_writer = tf.summary.FileWriter(Config.summary_log_dir, sess.graph)
# =============================================================================

    sess.run(init)

    for i in range(Config.training_iter):
        sess.run(train_model.optimizer)
    
# =============================================================================
#     if i % 100 == 0:
#         summary_str = sess.run(train_model.summary)
#         summary_writer.add_summary(summary_str, i)
#         summary_writer.flush()
# 
#     if (i + 1) % 1000 == 0 or (i + 1) == Config.training_iter:
#         checkpoint_file = os.path.join(Config.summary_log_dir, 'model.ckpt')
#         train_model.saver.save(sess, checkpoint_file, global_step=i)
# =============================================================================

    sv.saver.save(sess, 'model_save/')


# Evaluation
# train

    prd_len = 1000

    Train_Right = np.zeros(Config.output_size, dtype=np.float)
    Train_RMSE = np.zeros(Config.output_size, dtype=np.float)

    for i in range(prd_len):
        x, y, p = sess.run([train_model.input, train_model.valid, train_model.predict])
        x = x.reshape(-1)
        y = y.reshape(-1)
        p = p.reshape(-1)
        lax = x[-1]
        for j in range(Config.output_size):
            if ((y[j]-lax)*(p[j]-lax) > 0):
                Train_Right[j] += 1
            Train_RMSE[j] += pow(y[j]-p[j], 2)
    #print(x, y, p)

    print("Train_Right:", Train_Right/prd_len)
    print("Train_RMSE:", np.sqrt(Train_RMSE/prd_len))
    
    Test_Right = np.zeros(Config.output_size, dtype=np.float)
    Test_RMSE = np.zeros(Config.output_size, dtype=np.float)

    for i in range(prd_len):
        x, y, p = sess.run([test_model.input, test_model.valid, test_model.predict])
        x = x.reshape(-1)
        y = y.reshape(-1)
        p = p.reshape(-1)
        lax = x[-1]
        for j in range(Config.output_size):
            if ((y[j]-lax)*(p[j]-lax) > 0):
                Test_Right[j] += 1
            Test_RMSE[j] += pow(y[j]-p[j], 2)
    #print(x, y, p)

    print("Test_Right:", Test_Right/prd_len)
    print("Test_RMSE:", np.sqrt(Test_RMSE/prd_len))



