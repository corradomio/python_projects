import networkx as nx
import numpy as np


def is_symmetric(M):
    n = len(M)
    for i in range(n-1):
        for j in range(i+1, n):
            if M[i, j] != M[j, i]:
                return False
    return True


def draw(G, pos=None, ax=None, labels=True, **kwds):
    if isinstance(G, np.ndarray):
        M = G
        if is_symmetric(M):
            G = nx.Graph(M)
        else:
            G = nx.DiGraph(M)
    if pos is None:
        pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, ax=ax, **kwds)
    if labels:
        nx.draw_networkx_labels(G, pos=pos)
