# coding=utf-8
"""
不修改原数据，切分使用特征点
"""

from preprocessing.up2down import up2down
from sklearn.preprocessing import scale
from measurement.dtw import dtw
import numpy as np
import heapq
import matplotlib.pyplot as plt
import os
import pandas as pd


# ------- Config -------
data_dir = '../data/'
u2d_error = 0.02
u2d_split_length = 8
cutoff_present = 0.02
cluster_num = 10
# ----------------------

for path in os.listdir(data_dir):
    opst = os.path.splitext(path)
    if opst[1] == '.csv':
        price = pd.read_csv(os.path.join(data_dir, path))
        name = opst[0]

        adj_close = price.iloc[:, 5].values
        adj_close = scale(adj_close)
        u2d_adj_close_index = up2down(adj_close, u2d_error)

        barrel = []
        for index_ in range(len(u2d_adj_close_index) - u2d_split_length):
            feature_point_index_0, feature_point_index_1 = u2d_adj_close_index[index_], u2d_adj_close_index[
                index_ + u2d_split_length]
            feature_point_values = adj_close[feature_point_index_0: feature_point_index_1]
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

        c_rho = []
        c_delta = []

        mul_res = np.zeros(barrel_length, dtype=np.float)
        center_set = []

        for i in range(barrel_length):
            mul_res[i] = rho[i] * delta[i]
            top_k = heapq.nlargest(cluster_num, mul_res)

        print(name)
        # print(barrel_length)
        # print(rho)
        # print(delta)
        # print(mul_res)
        # print(topk)
        # print('\n')

        for i in range(barrel_length):
            if mul_res[i] in top_k:
                c_rho.append(rho[i])
                c_delta.append(delta[i])
                center_set.append(barrel[i])

        plt.scatter(rho, delta, c="k")
        plt.scatter(c_rho, c_delta, c="r", linewidths=2)
        plt.savefig('exp_res/'+name+'.png')
        plt.clf()

        center_avg_length = 0
        for c in center_set:
            center_avg_length += len(c)

        center_avg_length = round(center_avg_length/cluster_num)







