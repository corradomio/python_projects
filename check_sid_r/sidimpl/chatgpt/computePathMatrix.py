import numpy as np
import math


def compute_path_matrix(G):
    """
    Python version of the R function `computePathMatrix`.

    Parameters
    ----------
    G : array-like of shape (p, p)
        Adjacency matrix of a DAG.

    Returns
    -------
    np.ndarray of shape (p, p), dtype=bool
        Boolean path matrix where entry (i, j) is True if there is a
        directed path from i to j. The diagonal is also True.
    """
    G = np.asarray(G)
    p = G.shape[1]

    path_matrix = np.eye(p, dtype=int) + G

    k = math.ceil(math.log(p, 2)) if p > 1 else 0
    for _ in range(k):
        path_matrix = path_matrix @ path_matrix

    path_matrix = path_matrix > 0
    return path_matrix


def compute_path_matrix2(G, cond_set, path_matrix1):
    """
    Python version of the R function `computePathMatrix2`.

    Parameters
    ----------
    G : array-like of shape (p, p)
        Adjacency matrix of a DAG.
    cond_set : array-like
        Indices of nodes whose outgoing edges should be removed.
        Uses Python's 0-based indexing.
    path_matrix1 : np.ndarray
        Previously computed path matrix from `compute_path_matrix(G)`.

    Returns
    -------
    np.ndarray of shape (p, p), dtype=bool
        Boolean path matrix after removing all outgoing edges from cond_set.
        If cond_set is empty, returns path_matrix1.
    """
    G = np.asarray(G).copy()
    cond_set = np.asarray(cond_set, dtype=int)
    p = G.shape[1]

    if len(cond_set) > 0:
        G[cond_set, :] = 0

        path_matrix2 = np.eye(p, dtype=int) + G

        k = math.ceil(math.log(p, 2)) if p > 1 else 0
        for _ in range(k):
            path_matrix2 = path_matrix2 @ path_matrix2

        path_matrix2 = path_matrix2 > 0
    else:
        path_matrix2 = path_matrix1

    return path_matrix2


import numpy as np
import math


# def computePathMatrix(G):
#     G = np.asarray(G)
#     p = G.shape[1]
#
#     PathMatrix = np.eye(p, dtype=int) + G
#
#     k = math.ceil(math.log(p, 2)) if p > 1 else 0
#     for _ in range(k):
#         PathMatrix = PathMatrix @ PathMatrix
#
#     PathMatrix = PathMatrix > 0
#     return PathMatrix


# def computePathMatrix2(G, condSet, PathMatrix1):
#     G = np.asarray(G).copy()
#     condSet = np.asarray(condSet, dtype=int)
#     p = G.shape[1]
#
#     if len(condSet) > 0:
#         G[condSet, :] = 0
#
#         PathMatrix2 = np.eye(p, dtype=int) + G
#
#         k = math.ceil(math.log(p, 2)) if p > 1 else 0
#         for _ in range(k):
#             PathMatrix2 = PathMatrix2 @ PathMatrix2
#
#         PathMatrix2 = PathMatrix2 > 0
#     else:
#         PathMatrix2 = PathMatrix1
#
#     return PathMatrix2