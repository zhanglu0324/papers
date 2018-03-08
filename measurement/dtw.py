# the code is focus on the distance time seires.
# author: zhanglu
# initial date: 2018.01.16
# DTW: Dynamic Time Warping
 
import numpy as np
from scipy.spatial.distance import cdist


def dtw(Q, C, dist):
    m, n = len(Q), len(C)
    D = np.zeros([m+1, n+1], dtype=np.float)
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf

    D1 = D[1:, 1:]
    for i in range(m):
        for j in range(n):
            D1[i, j] = dist(Q[i], C[j])
    # D[1:, 1:] = cdist(Q, C, dist)
    C1 = D1.copy()

    for i in range(m):
        for j in range(n):
            D1[i, j] += min(D[i, j], D[i+1, j], D[i, j+1])

    if len(Q) == 1:
        path = np.zeros(len(C)), range(len(C))
    elif len(C) == 1:
        path = range(len(Q)), np.zeros(len(Q))
    else:
        path = _traceback(D)
    return D1[-1, -1] / sum(D1.shape), C1, D1, path


def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while i > 0 or j > 0:
        tmp = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if tmp == 0:
            i -= 1
            j -= 1
        elif tmp == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

