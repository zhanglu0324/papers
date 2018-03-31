# -*- coding: utf-8 -*-
# =============================================================================
# 对照实验：ARIMA模型
# 作者：张璐
# 时间：2018-03-22
# =============================================================================

from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


data_dir = '../data/'
name = 'VTI'
price = pd.read_csv(os.path.join(data_dir, name+'.csv'))
close_price = price.iloc[:, 4].values
close_price = scale(close_price)

date1 = 1501
trainlen = 100
prdlen=5

exp_total = np.ones(prdlen, dtype=np.float)
exp_right = np.ones(prdlen, dtype=np.float)
RMSE = np.ones(prdlen, dtype=np.float)

wrongs = 0

for i in range(400, 650):
    dta = close_price[i:i+trainlen]
    
    begin = dta[-1]
    
    dta = np.array(dta, dtype=np.float)
    dta = pd.Series(dta)
    
    try:
        dta.index = pd.Index(sm.tsa.datetools.dates_from_range(str(date1+i),str(date1+i+trainlen-1)))
        arma_mod20 = sm.tsa.ARIMA(dta,(3,1,2),freq='A').fit()
        predict_sunspots = arma_mod20.predict(str(date1+i+trainlen), str(date1+i+trainlen+prdlen-1), dynamic=True)
        res = predict_sunspots.values
        valid = close_price[i+trainlen:i+trainlen+prdlen]
# =============================================================================
#         print('predict:', predict_sunspots.values)
#         print('valid:', close_price[i+trainlen:i+trainlen+prdlen])
# =============================================================================
    
        for i in range(prdlen):
            exp_total[i] += 1
            RMSE[i] += pow(valid[i]-res[i], 2)
            if (res[i]-begin) * (valid[i]-begin) >= 0:
                exp_right[i] += 1
    except:
        wrongs += 1
        pass
    
# =============================================================================
#     dta.index = pd.Index(sm.tsa.datetools.dates_from_range(str(date1+i),str(date1+i+trainlen-1)))
#     arma_mod20 = sm.tsa.ARMA(dta,(3,0),freq='A').fit()
#     predict_sunspots = arma_mod20.predict(str(date1+i+trainlen), str(date1+i+trainlen+prdlen-1), dynamic=True)
#     res = predict_sunspots.values
#     valid = close_price[i+trainlen:i+trainlen+prdlen]
#     print('predict:', predict_sunspots.values)
#     print('valid:', close_price[i+trainlen:i+trainlen+prdlen])
#     
#     
#     
#     for i in range(prdlen):
#         exp_total[i] += 1
#         RMSE[i] += pow(valid[i]-res[i], 2)
#         if (res[i]-begin) * (valid[i]-begin) >= 0:
#             exp_right[i] += 1
# 
# =============================================================================
evaluation_res = exp_right/exp_total
print("P:", evaluation_res)
print("RMSE:", np.sqrt(RMSE/exp_total))
print("wrongs:", wrongs)
                
# =============================================================================
# # time and index
# train_start = 1
# train_end = train_start + int(adj_close_data_length*train_present)-1
# test_end = train_start + adj_close
# =============================================================================



