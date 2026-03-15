import numpy as np
from math import ceil, log


def compute_path_matrix2(G, condSet, PathMatrix1):
    """
    Python translation of the R function `computePathMatrix2`.

    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of shape (p, p).
    condSet : list[int] or np.ndarray
        Nodes whose outgoing edges are removed. Assumed 0-based.
    PathMatrix1 : np.ndarray
        Precomputed path matrix returned by compute_path_matrix.

    Returns
    -------
    np.ndarray
        Boolean path matrix after removing all outgoing edges from condSet.
        If condSet is empty, returns PathMatrix1.
    """

    G = np.array(G, copy=True)
    condSet = np.asarray(condSet, dtype=int)
    p = G.shape[1]

    if len(condSet) > 0:
        G[condSet, :] = 0

        PathMatrix2 = np.eye(p, dtype=int) + G

        k = ceil(log(p) / log(2))
        for _ in range(k):
            PathMatrix2 = PathMatrix2 @ PathMatrix2

        PathMatrix2 = PathMatrix2 > 0
    else:
        PathMatrix2 = PathMatrix1

    return PathMatrix2