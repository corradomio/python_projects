import numpy as np
import math

def compute_path_matrix(G):
    """
    Takes an adjacency matrix G from a DAG and computes a path matrix.
    Entry (i,j) == True means there is a directed path from i to j.
    The diagonal will also be True.
    """
    G = np.array(G)
    p = G.shape[1]

    path_matrix = np.eye(p) + G

    k = math.ceil(math.log(p) / math.log(2))
    for _ in range(k):
        path_matrix = path_matrix @ path_matrix

    return path_matrix > 0


def compute_path_matrix2(G, cond_set, path_matrix1):
    """
    Same as compute_path_matrix, but first removes all edges leaving nodes in cond_set.
    If cond_set is empty, returns path_matrix1 unchanged.
    """
    G = np.array(G)
    p = G.shape[1]

    if len(cond_set) > 0:
        G[cond_set, :] = np.zeros((len(cond_set), p))

        path_matrix2 = np.eye(p) + G

        k = math.ceil(math.log(p) / math.log(2))
        for _ in range(k):
            path_matrix2 = path_matrix2 @ path_matrix2

        return path_matrix2 > 0
    else:
        return path_matrix1