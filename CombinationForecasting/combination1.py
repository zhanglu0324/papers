# -*- coding: utf-8 -*-
# =============================================================================
# 重载组合模型
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


name = 'AAPL'
infile = open('rpd/'+name+'.txt', 'r')

center_length = int(infile.readline().strip())
centers = []
for i in range(center_length):
    tmp = infile.readline().strip().split(',')
    centers.append(np.array(tmp, dtype=np.float))
    
class Config(object):
    input_size = 12

