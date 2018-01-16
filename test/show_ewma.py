import pandas as pd
from preprocessing.ewma import ewma
import matplotlib.pyplot as plt

prices = pd.read_csv("../data/^GSPC.csv") 
adj_close = prices.iloc[:, 3]
index = prices.index

begin = 0
end = 100
adj_close = adj_close[begin:end]
index = index[begin:end]

ewma_adj_close = ewma(adj_close, 10, 0.2)

plt.plot(index, adj_close)
plt.plot(index, ewma_adj_close)
plt.show()
