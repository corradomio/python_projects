from typing import Union

import numpy as np
import networkx as nx


def d_separation(adj_matrix):
    n = adj_matrix.shape[0]
    d_sep_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Marking all pairs as d-separated initially
                d_sep_matrix[i, j] = 1

                # Performing depth-first search from node i to check if node j is reachable
                visited = set()
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node == j:
                        d_sep_matrix[i, j] = 0
                        break
                    visited.add(node)
                    parents = np.where(adj_matrix[:, node] != 0)[0]
                    for parent in parents:
                        if parent not in visited:
                            stack.append(parent)
    return d_sep_matrix.astype(np.int8)
# end


def d_separation_pairs(G: Union[nx.DiGraph, np.ndarray]) -> np.ndarray:
    """
    Compute the 'd-separation' matrix based on networkx 'is_d_separated()'
    applied to each node pairs
    :param G: digraph or adjacency matrix
    :return: 'd-separation' matrix
    """
    if isinstance(G, np.ndarray):
        A: np.ndarray = G
        G = nx.from_numpy_array(A, create_using=nx.DiGraph())
    n = G.number_of_nodes()
    mat = np.zeros((n, n), dtype=np.int8)
    empty = set()

    for u in range(n-1):
        for v in range(u+1, n):
            try:
                dsep = nx.is_d_separator(G, u, v, empty)
            except:
                dsep = 0
            mat[u, v] = dsep
            mat[v, u] = dsep
    # end u/v
    return mat
# end


def power_adjacency_matrix(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    I = np.identity(n, int)
    A = I + A

    P = I
    for e in range(n-1):
        P = np.dot(P, A)
    P[P > 0] = 1
    P = P.astype(A.dtype)
    return P
