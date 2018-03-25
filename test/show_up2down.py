import pandas as pd
from preprocessing.up2down import up2down
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

prices = pd.read_csv("../data/bak/^GSPC.csv") 
adj_close = prices.iloc[:, 3]
index = prices.index

begin = 0
end = 100
adj_close = adj_close[begin:end]
#adj_close = scale(adj_close)
index = index[begin:end]

pos = up2down(adj_close, 0.01)
sec_ac = []
for i in pos:
    sec_ac.append(adj_close[i])

plt.figure(figsize=(6,4))
plt.plot(index, list(map(lambda x:x+40, adj_close)), label="Raw", c="steelblue")
plt.plot(pos, sec_ac, "x-", label="PLR", c="darkred")
plt.xlabel("Days")
plt.ylabel("Close price ")
plt.legend()
plt.savefig("test_figure/up2down.pdf")
