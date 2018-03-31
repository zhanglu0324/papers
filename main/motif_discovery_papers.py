# coding=utf-8
"""
不修改原数据，切分使用特征点
2018-03-19：统一cutoff_distance与delta阈值，设立补集概念
"""

from preprocessing.up2down import up2down
from sklearn.preprocessing import scale
from measurement.dtw import dtw
from measurement.mvm import minimal_variance_matching
import numpy as np
import heapq
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
       
        
        cluster_num = len(center_set)

        plt.scatter(rho, delta, c="k", label=r'$\complement_S R$')
        plt.scatter(c_rho, c_delta, c="r", label=r'$R$')
        plt.plot([rho_threshold]*2,[0,max(delta)], '--', c='b')
        plt.annotate(r'$\rho_c$',xy=(rho_threshold, max(delta)*0.6),
                     xytext=(rho_threshold*1.3, max(delta)*0.8), 
                     arrowprops=dict(facecolor='k', headlength=10, headwidth=5, width=1))
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$\delta$')
        plt.legend()
        plt.savefig('exp_res/'+name+'.pdf')
        plt.clf()


        print(name+': '+str(cluster_num))
# =============================================================================
#         print(barrel_length)
#         print(rho)
#         print(delta)
#         print(mul_res)
#         print(topk)
#         print('\n')
# =============================================================================

              
        # ------- Prediction -------
        split_length = u2d_split_length
        
        exp_total = np.ones(prediction_step_length, dtype=np.float)
        exp_right = np.ones(prediction_step_length, dtype=np.float)
        RMSE = np.ones(prediction_step_length, dtype=np.float)
        exp_total_count = 0
        exp_miss_count = 0
        
        debug_cnt = 0
        for i in range(len(test_data) - split_length):
            segment_tmp = test_data[i: i+split_length]
            local_segment = segment_tmp[:u2d_split_length-prediction_step_length]
            valid_segment = segment_tmp[u2d_split_length-prediction_step_length:]
            
            local_segment = scale(local_segment)
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
            
            # TODO: evaluation
# =============================================================================
#             print("res:", res)
#             print("valid:", valid_segment)
# =============================================================================
            last_x = local_segment[-1]
            
            for i in range(prediction_step_length):
                if (res[i]-last_x) != 0:
                    exp_total[i] += 1
                    RMSE[i] += pow(valid_segment[i]-res[i], 2)
                    if (res[i]-last_x) * (valid_segment[i]-last_x) > 0:
                        exp_right[i] += 1
                else:
                    exp_miss_count += 1
                    break
            exp_total_count += 1
            
        
# =============================================================================
#             # draw picture with res
#             plt.figure(figsize=(6, 4))
#             Y2 = list(map(lambda x:x+3, mvm_target))
#             X2 = list(range(len(mvm_target))) 
#             plt.plot(X2, Y2, '-x', c='k', label='target')
#             path_0, path_1 = list(zip(*mvm_target_cor))
#             X1 = list(map(lambda x:x+(path_1[-1]+path_1[0]-split_length)/2, path_0))
#             plt.plot(X1, local_segment, '->', c='r', label='query')
#             for i in range(len(local_segment)):
#                 plt.plot([X1[i], path_1[i]], [local_segment[i], Y2[path_1[i]]], '--', c='k')
#             
#             X3 = [i+X1[-1] for i in range(prediction_step_length+1)]
#             Y3 = np.append(local_segment[-1], res)
#             Y4 = np.append(local_segment[-1], valid_segment)
#             plt.plot(X3, Y3, '-o', c='b', label='prediction')
#             plt.plot(X3, Y4, '-*', c='g', label='valid')
#             plt.xlabel('Days')
#             plt.ylabel('Normalized close price')
#             plt.legend()
#             plt.savefig('mvm_res/'+name+str(debug_cnt)+'.pdf')
#             plt.clf()
#             
#             debug_cnt += 1
#             if debug_cnt > 15:
#                 break
# =============================================================================
            
        evaluation_res = exp_right/exp_total
        print("P:", evaluation_res)
        print("RMSE:", np.sqrt(RMSE/exp_total))
        print("miss rate: ", exp_miss_count*1.0/exp_total_count)
# =============================================================================
#             # draw picture
#             plt.figure(figsize=(10, 8))
#             Y2 = list(map(lambda x:x+3, mvm_target))
#             X2 = list(range(len(mvm_target))) 
#             plt.plot(X2, Y2, '-x', c='k', label='target subseqence')
#             path_0, path_1 = list(zip(*mvm_target_cor))
#             X1 = list(map(lambda x:x+(path_1[-1]+path_1[0]-split_length)/2, path_0))
#             plt.plot(X1, local_segment, '->', c='r', label='query subseqence')
#             for i in range(len(local_segment)):
#                 plt.plot([X1[i], path_1[i]], [local_segment[i], Y2[path_1[i]]], '--', c='k')
#             plt.legend()
#             plt.savefig('mvm_res/'+name+str(debug_cnt)+'.png')
#             plt.clf()
# =============================================================================
            
            
            
            
            
            