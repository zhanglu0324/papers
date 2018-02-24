import os
import pandas as pd

def data_loader(dir):
    prices = []

    for path in os.listdir(dir):
        if os.path.splitext(path)[1] == '.csv':
            price = pd.read_csv(os.path.join(dir, path))
            prices.append(price)

    return prices