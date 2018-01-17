# compare dtw: 2018-01-17

import pandas as pd
from measurement.dtw import DTW
import matplotlib.pyplot as plt
import numpy as np

prices = pd.read_csv("../data/^GSPC.csv") 
adj_close = prices.iloc[:, 3]
index = prices.index

start = 0
offset = 100
a1 = adj_close[start: start+offset].values

start = 300
a2 = adj_close[start: start+offset].values

dist = lambda x, y: np.linalg.norm(np.array(x)-np.array(y), ord=2)

res, path, acc = DTW(a1, a2, fun=dist)
print(res)




plt.plot(a1)
plt.plot(a2)
plt.show()





