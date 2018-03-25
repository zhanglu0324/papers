# -*- coding: utf-8 -*-
# =============================================================================
# 组合模型行为模式获取
# 作者：张璐
# 时间：2018-03-26
# =============================================================================

from preprocessing.up2down import up2down
from sklearn.preprocessing import scale
from measurement.dtw import dtw
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# ------- Config -------
data_dir = '../data/'
u2d_error = 0.02
u2d_split_length = 12
cutoff_present = 0.10
train_present = 0.8
test_present = 0.2
prediction_step_length = 5
# ----------------------

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
        
        # ------- Pattern Discovery -------
        u2d_adj_close_index = up2down(train_data, u2d_error)

        barrel = []
        for index_ in range(len(u2d_adj_close_index) - u2d_split_length):
            feature_point_index_0, feature_point_index_1 = u2d_adj_close_index[index_], u2d_adj_close_index[
                index_ + u2d_split_length]
            feature_point_values = train_data[feature_point_index_0: feature_point_index_1]
            barrel.append(feature_point_values)

        barrel_length = len(barrel)
        nn_distance = np.zeros([barrel_length, barrel_length], dtype=np.float)
        distance_list = []
        for i_ in range(barrel_length - 1):
            for j_ in range(i_ + 1, barrel_length):
                dist, cost, acc, path = dtw(barrel[i_], barrel[j_], dist=lambda x, y: abs(x - y))
                nn_distance[i_, j_] = dist
                nn_distance[j_, i_] = dist
                distance_list.append(dist)
        distance_array = np.array(distance_list)

        cutoff_position = int(len(distance_array) * cutoff_present)
        sorted_arr = np.sort(distance_array)
        cutoff_distance = sorted_arr[cutoff_position]

        rho = np.zeros(barrel_length, dtype=np.float)
        for i in range(barrel_length - 1):
            for j in range(i, barrel_length):
                dist = np.exp(-(nn_distance[i][j] / cutoff_distance) ** 2)
                rho[i] += dist
                rho[j] += dist

        delta = np.ones(barrel_length, dtype=np.float) * 10000
        max_density = np.max(rho)
        for i in range(barrel_length):
            if rho[i] < max_density:
                for j in range(barrel_length):
                    if rho[i] < rho[j] and delta[i] > nn_distance[i][j]:
                        delta[i] = nn_distance[i][j]
            else:
                delta[i] = 0.0
                for j in range(barrel_length):
                    if delta[i] < nn_distance[i][j]:
                        delta[i] = nn_distance[i][j]

        rho_threshold = max(rho)/3.0
        
        c_rho = []
        c_delta = []
        center_set = []
        
# =============================================================================
#         for i in range(barrel_length):
#             if rho[i]>rho_threshold and delta[i]>cutoff_distance:
#                 c_rho.append(rho[i])
#                 c_delta.append(delta[i])
#                 center_tmp = scale(barrel[i])
#                 center_set.append(center_tmp)
# =============================================================================
        
        center_index = [np.argmax(rho)]
        sorted_rho = list(rho)
        sorted_rho.sort(reverse=True)
        for i in range(1, barrel_length):
            if sorted_rho[i] > rho_threshold:
                flag = 1
                this_index = list(rho).index(sorted_rho[i])
                for tmp_index in center_index:
                    if nn_distance[tmp_index][this_index] < cutoff_distance:
                        flag = 0
                        break
                if flag:
                    center_index.append(this_index)
        
        for tmp_index in center_index:
            center_set.append(scale(barrel[tmp_index]))
            c_rho.append(rho[tmp_index])
            c_delta.append(delta[tmp_index])
        
        outfile = open('rpd/'+name+'.txt', 'w')
        outfile.write(str(len(center_set))+'\n')
        for center in center_set:
            tmp = ','.join(map(str, center))
            outfile.write(tmp+'\n')
        outfile.close()