# the code is focus on main trend of time seires.
# author: zhanglu
# initial date: 2018.01.15
# EWMA: Exponentially Weighted Moving Average

def ewma(seires, w):
    res = []
    for i in range(len(seires)):
        if i < w - 1:
            
