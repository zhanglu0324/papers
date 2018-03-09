# =============================================================================
# useful code: apple -> AAPL, S&P 500 -> ^GSPC 
# =============================================================================

import pandas_datareader.data as web
from datetime import datetime

def web_reader(code, start, end):
    return web.DataReader(code, 'iex', start, end)


codes = ['SPY', 'AAPL', 'VTI']
for code in codes:
    prices = web_reader(code, datetime(2014, 1, 1), datetime(2018, 3, 1))
    prices.to_csv(code+'.csv')

