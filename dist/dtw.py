# the code is focus on the distance time seires.
# author: zhanglu
# initial date: 2018.01.16
# DTW: Dynamic Time Warping
 
import numpy as np


def AccumulatedCostMatrix(Q, C, fun):
    n = len(Q)
    m = len(C)
    
    dist = np.zeros([n, m], dtype=np.float)
    
    for i in range(n):
        for j in range(m):
            dist[i][j] = fun(Q[i], C[j])
    
    dtw = np.zeros([n, m], dtype=np.float)
    dtw[0][0] = dist[0][0]
    
    for i in range(1, n):
        dtw[i][0] = dtw[i-1][0] + dist[i][0]
    
    for j in range(1, m):
        dtw[0][j] = dtw[0][j-1] + dist[0][j]
    
    for i in range(1, n):
        for j in range(1, m):
            dtw[i][j] = dist[i][j] + min([dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1]])
    
    return dtw


def OptimalWarpingPath(dtw):
    
# def dtw(Q, C):
    