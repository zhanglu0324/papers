# the code is focus on main trend of time seires.
# author: zhanglu
# initial date: 2018.01.15
# EWMA: Exponentially Weighted Moving Average

def ewma(seires, w, alpha):
    res = []
    for i in range(len(seires)):
        cnt = 0
        up = 0.0
        down = 0.0
        while cnt < w and i-cnt >= 0:
            up += pow(1-alpha, cnt) * seires[i-cnt]
            down += pow(1-alpha, cnt)
            cnt += 1
        res.append(up/down)
    return res
            
            
            
            
