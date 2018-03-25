# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 07:46:21 2018

@author: zhanglu
"""

#coding=utf-8
# =============================================================================
# 绘制PPEM、LSTM、ARIMA三种算法MSE的情况
# 作者：张璐
# 日期：2018-03-26
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


AAPL_LSTM1_MSE = [0.571, 0.561, 0.556, 0.554, 0.526]
AAPL_LSTM2_MSE = [0.607, 0.589, 0.582, 0.58, 0.565]
AAPL_PPEM_MSE = [ 0.61910995, 0.59366755, 0.58970976, 0.56728232, 0.53298153]
AAPL_ARIMA_MSE = [ 0.57142857, 0.53896104, 0.53246753, 0.52597403, 0.54142857]
AAPL = [ AAPL_LSTM2_MSE,AAPL_LSTM1_MSE, AAPL_PPEM_MSE, AAPL_ARIMA_MSE]


color_index = ['slategray', 'g', '#d62728', 'steelblue']
label_index = ['LSTM1', 'LSTM2', 'PPEM', 'ARIMA']
ind = np.arange(5) 
          
fig = plt.figure(figsize=(6,4))         
ax1 = fig.add_subplot(111)

width = 0.18
for i in range(4):
    ax1.bar(ind+i*width+0.5, AAPL[i], width=width,
            color=color_index[i], label=label_index[i])
ax1.set_title('AAPL')
ax1.set_ylabel('Precision') 
ax1.set_xlabel('Days')
ax1.set_ylim(0.45, 0.7)
plt.legend()
plt.savefig('pictures/AAPL_M3_TA.pdf')