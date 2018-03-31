# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

def minimal_variance_matching(query, target, skip_elements=2):
    m = len(query)
    n = len(target)
    r = np.zeros([m, n], dtype=np.float)
    
    for i in range(m):
        for j in range(n):
            r[i, j] = abs(query[i] - target[j])
    
    # print(r)
    
    # 建立初始图
    G = nx.DiGraph()
    
    # 建立映射字典 与 反转字典
    dic = {}
    cnt = 0
    for i in range(m):
        for j in range(i, i+n-m+1):
            dic[cnt] = (i, j)
            # G.add_node(cnt)
            cnt += 1
            
    r_dic = dict(zip(dic.values(), dic.keys()))
    
    # 加入边，-1代表起始点，-2代表终点
    for i in range(n-m+1):
        G.add_edge(-1, r_dic[(0, i)], weight=r[0, i])
    
    for i in range(1, m):
        for j in range(i, i+n-m+1):
            start_pos = max(i-1, j-skip_elements-1)
            for k in range(start_pos, j):
                G.add_edge(r_dic[(i-1, k)], r_dic[(i, j)], weight=r[i, j])
    
    for i in range(m-1, n):
        G.add_edge(r_dic[(m-1, i)], -2, weight=0)
    
    path = nx.dijkstra_path(G, source=-1, target=-2)
    
    correspondence = []
    
    for i in range(len(path) - 2):
        correspondence.append(dic[path[i+1]])
    
    distance=nx.dijkstra_path_length(G, source=-1, target=-2)
    
    return correspondence, distance


