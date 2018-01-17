# compare dtw: 2018-01-17

import pandas as pd
from measurement.dtw import DTW
import matplotlib.pyplot as plt
import numpy as np
import dtw as thedtw

prices = pd.read_csv("../data/^GSPC.csv") 
adj_close = prices.iloc[:, 3]
index = prices.index

start = 0
offset = 100
a1 = adj_close[start: start+offset].values

start = 300
a2 = adj_close[start: start+offset].values

dist = lambda x, y: np.linalg.norm(np.array(x)-np.array(y), ord=2)

# res, path, acc = DTW(a1, a2, fun=dist)
res, path, acc = DTW(np.array(a1).reshape(-1, 1), np.array(a2).reshape(-1, 1), fun=lambda x, y: np.linalg.norm(x-y, ord=2))

print("res1", res)

dist2, cost2, acc2, path2 = thedtw.dtw(np.array(a1).reshape(-1, 1), np.array(a2).reshape(-1, 1), dist=lambda x, y: np.linalg.norm(x - y, ord=2))

print("res2", dist2)
print("acc1")
print(acc)
print("acc2")
print(acc2)

print("path1")
print(path)
print("path2")
print(path2)

plt.plot(a1)
plt.plot(a2)
plt.show()





