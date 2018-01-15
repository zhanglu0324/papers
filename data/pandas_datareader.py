# =============================================================================
# get data from yahoo finance
# author: zhanglu
# initial time: 2018-01-15
# useful code: apple -> AAPL, S&P500 -> `SPX 
# =============================================================================

import pandas_datareader.data as web

def web_reader(code, start, end):
    return web.DataReader(code, 'yahoo', start, end)