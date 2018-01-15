# =============================================================================
# useful code: apple -> AAPL, S&P 500 -> ^GSPC 
# =============================================================================
from data.pandas_datareader import web_reader
from datetime import datetime

codes = ['^GSPC', 'AAPL', '^DJI']
for code in codes:
    prices = web_reader(code, datetime(2015, 1, 1), datetime(2017, 12, 30))
    prices.to_csv(code+'.csv')

