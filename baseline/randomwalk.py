# -*- coding: utf-8 -*-
# =============================================================================
# 对照试验1，随机游走模型
# 作者：张璐
# 日期：2018-03-21
# =============================================================================


from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


data_dir = '../data/'
predict_length = 5
train_present = 0.8
test_present = 0.2
exp_cnt = 100


for path in os.listdir(data_dir):
    opst = os.path.splitext(path)
    if opst[1] == '.csv':
        price = pd.read_csv(os.path.join(data_dir, path))
        name = opst[0]

        adj_close = price.iloc[:, 4].values
        adj_close = scale(adj_close)
        
        adj_close_data_length = len(adj_close)
        
        tt_threshold = int(adj_close_data_length * train_present)
        train_data = adj_close[:tt_threshold]
        test_data = adj_close[tt_threshold:]
        
        # train
        std = np.std(train_data)
        
        # prediction
        exp_total = np.ones(predict_length, dtype=np.float)
        exp_right = np.ones(predict_length, dtype=np.float)
        RMSE = np.ones(predict_length, dtype=np.float)
        
        for i in range(len(test_data) - predict_length - 1):
            segment = test_data[i: i+predict_length + 1]
            begin = segment[0]
            valid = segment[1:]
            
            for i in range(exp_cnt):
                curr = begin
                res = np.zeros(predict_length, dtype=np.float)
                for i in range(predict_length):
                    curr = curr + np.random.normal(loc=0, scale=std)
                    res[i] = curr
                
            
                for i in range(predict_length):
                    exp_total[i] += 1
                    RMSE[i] += pow(valid[i]-res[i], 2)
                    if (res[i]-begin) * (valid[i]-begin) >= 0:
                        exp_right[i] += 1
            
            
        print(name)
        print("std:", std)
        evaluation_res = exp_right/exp_total
        print("P:", evaluation_res)
        print("RMSE:", np.sqrt(RMSE/exp_total))
                
                
                
                
                
                
                
                
                