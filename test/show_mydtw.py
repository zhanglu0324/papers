# compare dtw: 2018-01-17

import pandas as pd
from measurement.dtw import dtw
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
res, cost, acc, path = dtw(np.array(a1).reshape(-1, 1), np.array(a2).reshape(-1, 1), dist=lambda x, y: np.linalg.norm(x-y, ord=2))

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

# cmaps index
cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
         ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
         ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper']),
         ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
         ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c']),
         ('Miscellaneous', [
            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
ax1.imshow(acc.T, origin='lower', interpolation='nearest')
new_path = path
ax1.plot(new_path[0], new_path[1], 'w')
# ax1.set_xlim((-0.5, acc.shape[0]-0.5))
# ax1.set_ylim((-0.5, acc.shape[1]-0.5))
ax2.imshow(acc2.T, origin='lower', interpolation='nearest')
ax2.plot(path2[0], path2[1], 'w')

plt.show()






