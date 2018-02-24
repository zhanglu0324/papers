from data.data_loader import data_loader
from preprocessing.up2down import up2down
from sklearn.preprocessing import scale
import numpy as np


prices = data_loader('../data/')

# ------- Config -------
u2d_error = 0.02
u2d_split_length = 10
# ----------------------

for price in prices:
    adj_close = price.iloc[:, 5].values
    adj_close = scale(adj_close)
    u2d_adj_close_index = up2down(adj_close, u2d_error)

    barrel = []
    for index_ in range(len(u2d_adj_close_index) - u2d_split_length):
        feature_point_index = u2d_adj_close_index[index_: u2d_split_length]
        feature_point_values = [0] * u2d_split_length
        for i_ in range(u2d_split_length):
            feature_point_values[i_] = adj_close[feature_point_index[i_]]
        barrel.append((feature_point_index, feature_point_values))

    barrel_length = len(barrel)
    nn_distance = np.zeros([barrel_length, barrel_length], dtype=np.float)
    for i_ in range(barrel_length - 1):
        for j_ in range(i_+1, barrel_length):
            pass



    # print(len(adj_close))
    # print(len(up2down(adj_close, u2d_error)))

