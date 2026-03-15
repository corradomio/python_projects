import numpy as np
from math import ceil, log


def compute_path_matrix(G):
    """
    Python translation of the R function `computePathMatrix`.

    Parameters
    ----------
    G : np.ndarray
        Adjacency matrix of a DAG (shape p x p).

    Returns
    -------
    np.ndarray (bool)
        Path matrix where entry (i, j) is True if there is a directed
        path from i to j. Diagonal entries are also True.
    """

    G = np.asarray(G)
    p = G.shape[1]

    PathMatrix = np.eye(p, dtype=int) + G

    k = ceil(log(p) / log(2))

    for _ in range(k):
        PathMatrix = PathMatrix @ PathMatrix

    # Convert to boolean matrix indicating reachability
    PathMatrix = PathMatrix > 0

    return PathMatrix