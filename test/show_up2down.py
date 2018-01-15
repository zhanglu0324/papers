import pandas as pd
from preprocessing.up2down import up2down
import matplotlib.pyplot as plt

prices = pd.read_csv("../data/^GSPC.csv") 
adj_close = prices.iloc[:, 3]
index = prices.index

begin = 0
end = 100
adj_close = adj_close[begin:end]
index = index[begin:end]

pos = up2down(adj_close, 0.02)
sec_ac = []
for i in pos:
    sec_ac.append(adj_close[i])

plt.plot(index, adj_close)
plt.plot(pos, sec_ac)
plt.show()