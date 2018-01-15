# the code is focus on main trend of time seires.
# author: zhanglu
# initial date: 2018.01.15

import math

class point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

def point2line(p1, p2, p0):
    top = abs((p1.x-p0.x)*(p2.y-p1.y)+(p1.y-p0.y)*(p1.x-p2.x))
    down = math.sqrt(pow(p2.y-p1.y, 2)+pow(p2.x-p1.x, 2))
    return top/down     
    
def up2down(seires, r):
    ll = len(seires)
    points = []
    for i in range(0, ll):
        points.append(point(i, seires[i]))
    
    res = [0, ll-1]
    
    r = (max(seires) - min(seires)) * r
    def emax(begin, end):
        distance = []
        dist = lambda x: point2line(points[begin], points[end], x)
        for i in range(begin+1, end):
            distance.append(dist(points[i]))
        if max(distance) > r:
            k = begin + 1 + distance.index(max(distance))
            res.append(k)
            if k - begin > 1:
                emax(begin, k)
            if end - k > 1:
                emax(k, end)
    
    emax(0, ll-1)
    res.sort()
    return res

