import pandas as pd
from preprocessing.ewma import ewma
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates

prices = pd.read_csv("../data/^GSPC.csv")
adj_close = prices.iloc[:, 3]
index = prices.index

begin = 0
end = 100
adj_close = adj_close[begin:end]
index = index[begin:end]

ewma_adj_close = ewma(adj_close, 10, 0.2)


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(index, adj_close, label="adj_close", c="k")
ax1.plot(index, ewma_adj_close, label="EWMA", c="r")

# set x axis format
# xmajorLocator = MultipleLocator(5)
#
# ax1.xaxis.set_major_locator(xmajorLocator)
# ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
# plt.xticks(index, rotation=20)
# for label in ax1.xaxis.get_ticklabels():
#     label.set_rotation(20)

plt.legend()
plt.show()
