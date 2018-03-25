# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 07:46:21 2018

@author: zhanglu
"""

#coding=utf-8
# =============================================================================
# 绘制PPEM、LSTM、ARIMA三种算法RMSE的情况
# 作者：张璐
# 日期：2018-03-26
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


AAPL_LSTM1_MSE = [0.587, 0.610, 0.560, 0.542, 0.570]
AAPL_LSTM2_MSE = [0.789, 0.788, 0.786, 0.783, 0.781]
AAPL_PPEM_MSE = [1.53016907, 1.59019494, 1.64016952, 1.70151097, 1.83819645]
AAPL_ARIMA_MSE = [0.30467381, 0.30655205, 0.30954978, 0.30955815, 0.30818355]
AAPL = [AAPL_LSTM2_MSE, AAPL_LSTM1_MSE, AAPL_PPEM_MSE, AAPL_ARIMA_MSE]


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
ax1.set_ylabel('RMSE') 
ax1.set_xlabel('Days')
ax1.set_ylim(0, 2.5)
plt.legend()
plt.savefig('pictures/AAPL_M3_MSE.pdf')