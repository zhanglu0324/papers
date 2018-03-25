#coding=utf-8
# =============================================================================
# 绘制PPEM、Random Walker、ARIMA三种算法MSE的情况
# 作者：张璐
# 日期：2018-03-22
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt


AAPL_PPEM_MSE = [1.53016907, 1.59019494, 1.64016952, 1.70151097, 1.83819645]
AAPL_RW_MSE = [0.65209734, 0.92305015, 1.1275418, 1.29882752, 1.45364402]
AAPL_ARIMA_MSE = [0.30467381, 0.30655205, 0.30954978, 0.30955815, 0.30818355]
AAPL = [AAPL_PPEM_MSE, AAPL_RW_MSE, AAPL_ARIMA_MSE]

SPY_PPEM_MSE = [1.34957775, 1.32469241, 1.28859691, 1.3335859, 1.34776855]
SPY_RW_MSE = [0.57545644, 0.81811637, 0.99899502, 1.15788905, 1.2960299]
SPY_ARIMA_MSE = [0.1172826, 0.15158676, 0.1738969, 0.18916294, 0.19600753]
SPY = [SPY_PPEM_MSE, SPY_RW_MSE, SPY_ARIMA_MSE]

VTI_PPEM_MSE = [1.48239149, 1.50415778, 1.53612738, 1.54944678, 1.55023275]
VTI_RW_MSE = [0.58748293, 0.82320526, 1.00312294, 1.1519131, 1.28471384]
VTI_ARIMA_MSE = [0.11711435, 0.15188506, 0.17542316, 0.19183575, 0.19983311]
VTI = [VTI_PPEM_MSE, VTI_RW_MSE, VTI_ARIMA_MSE]


color_index = ['#d62728', 'g', 'steelblue']
label_index = ['PPEM', 'Random Walker', 'ARIMA']
ind = np.arange(5) 
          
fig = plt.figure(figsize=(6,4))         
ax1 = fig.add_subplot(111)

width = 0.25
for i in range(3):
    ax1.bar(ind+i*width+0.5, AAPL[i], width=width,
            color=color_index[i], label=label_index[i])
ax1.set_title('AAPL')
ax1.set_ylabel('RMSE') 
ax1.set_xlabel('Days')
ax1.set_ylim(0, 2.5)
plt.legend()
plt.savefig('pictures/AAPL_MSE.pdf')
# =============================================================================
# #plt.clf()
# 
# for i in range(3):
#     ax1.bar(ind+i*width+0.5, SPY[i], width=width,
#             color=color_index[i], label=label_index[i])
# ax1.set_title('SPY')
# ax1.set_ylabel('RMSE') 
# ax1.set_ylim(0, 2.5)
# plt.legend()
# plt.savefig('pictures/SPY.pdf')
# #plt.clf()              
# 
# 
# for i in range(3):
#     ax1.bar(ind+i*width+0.5, VTI[i], width=width,
#             color=color_index[i], label=label_index[i])
# ax1.set_title('VTI')
# ax1.set_ylabel('RMSE') 
# ax1.set_ylim(0, 2.5)
# plt.legend()
# plt.savefig('pictures/VTI.pdf')
# #plt.clf()               
# =============================================================================
