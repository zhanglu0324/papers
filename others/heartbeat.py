import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("heartpoint.csv")
series = df.iloc[:, 1].values[100:1100]

i = 0
res = []
while i < len(series):
    res.append(sum(map(float, series[i: i + 10]))/10)
    i += 10

print(" ".join(map(str, res)))
plt.plot(res)
plt.show()
