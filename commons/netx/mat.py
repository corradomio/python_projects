
__all__ = [
    "is_symmetric",
    "from_adjacency_matrix",
    "from_numpy_matrix",
    "from_numpy_array",
    "adjacency_matrix",
    "power_adjacency_matrix",
    "is_empty_adjacency_matrix",
    "is_full_adjacency_matrix",
]

from typing import Iterator

import networkx as nx
import numpy as np
from .graph import Graph
from scipy.sparse import csr_array
from random import randrange


# ---------------------------------------------------------------------------
# is_symmetric
# ---------------------------------------------------------------------------

def ilog2(x):
    i, e = 0, 1
    while e < x:
        e *= 2
        i += 1
    return i

def _rand_int_iter(ulim: int) -> Iterator[int]:
    while True:
        yield randrange(ulim)


def _bools(nbits: int, iseq: Iterator[int]):
    for ival in iseq:
        bits = [False]*nbits
        m = 1
        for j in range(nbits):
            if ival & m:
                bits[j] = True
            m <<= 1
        yield bits


# ---------------------------------------------------------------------------
# is_symmetric
# ---------------------------------------------------------------------------

def is_symmetric(M: np.ndarray):
    n = len(M)
    for i in range(n-1):
        for j in range(i+1, n):
            if M[i, j] != M[j, i]:
                return False
    return True
# end

# ---------------------------------------------------------------------------
# Adjacency matrix -> Graph
# ---------------------------------------------------------------------------

def from_adjacency_matrix(adjacency_matrix: np.ndarray, create_using=None) -> Graph:
    if isinstance(adjacency_matrix, csr_array):
        adjacency_matrix = adjacency_matrix.toarray()

    assert isinstance(adjacency_matrix, np.ndarray)
    n: int = adjacency_matrix.shape[0]

    def is_direct():
        for i in range(n):
            for j in range(i+1,n):
                if adjacency_matrix[i,j] != adjacency_matrix[j, i]:
                    return True
        return False

    def has_loops():
        for i in range(n):
            if adjacency_matrix[i,i] == 1:
                return True
        return False

    # if create_using is None:
    #     G = Graph(direct=is_direct(), loops=has_loops(), multi=False, acyclic=False)
    # else:
    #     G = create_using()

    if create_using and create_using == type(Graph):
        G = create_using(direct=is_direct(), loops=has_loops(), multi=False, acyclic=False)
    elif create_using:
        G = create_using()
    elif is_direct():
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # add nodes
    G.add_nodes_from(range(n))

    # add edges
    for u in range(n):
        for v in range(n):
            if adjacency_matrix[u,v] == 1:
                G.add_edge(u, v)

    return G
# end


# networkx v2.X method to create a graph from an adjacency matrix
def from_numpy_matrix(adjacency_matrix: np.ndarray, create_using=None) -> Graph:
    return from_adjacency_matrix(adjacency_matrix, create_using=create_using)
# end


# networkx v3.X method to create a graph from an adjacency matrix
def from_numpy_array(adjacency_matrix: np.ndarray, create_using=None) -> Graph:
    # parallel_edges=False, edge_attr='weight', *, nodelist=None
    return from_adjacency_matrix(adjacency_matrix, create_using=create_using)
# end


# ---------------------------------------------------------------------------
# adjacency_matrix
# power_adjacency_matrix
# is_empty_adjacency_matrix
# ---------------------------------------------------------------------------
# direct=False,
# loops=False,
# acyclic=False,
# multi=False,

def adjacency_matrix(G: nx.Graph, dtype=np.int8) -> np.ndarray:
    """
    Create the adjacency matrix [0,1] from the graph G.
    If the graph is undirected, the matrix is symmetric.

    TODO: it doesnt support NOT integer nodes!

    :param G: graph
    :param dtype: matrix elements' type
    :return: adjacency matrix
    """
    n = G.order()
    is_directed = G.is_directed()
    A = np.zeros((n, n), dtype=dtype)
    for u, v in G.edges():
        w = 1
        A[u, v] = w
        if not is_directed:
            A[v, u] = w
    # end
    return A
# end


def power_adjacency_matrix(A: np.ndarray, exp=-1) -> np.ndarray:
    """
    Compute the power of an adjacency matrix.

    :param A: adiacency matrix
    :param exp: exponent of the power. If -1, the exponent is computed as ilog2(n)
    :param condset: conditional set. Clear the output edges of the set
    :param loop: if to include the loops (diagonal=1) in the result matrix
    :return:
    """

    n, _ = A.shape
    I = np.identity(n, int)

    A = I + A

    if exp >= 0:
        P = I
        for e in range(exp):
            P = np.dot(P, A)
    else:
        l = ilog2(n)
        P = A
        for e in range(l):
            P = np.dot(P, P)

    P[P > 0] = 1

    return P
# end


def is_empty_adjacency_matrix(A: np.ndarray) -> bool:
    return A.sum() == 0


def is_full_adjacency_matrix(A: np.ndarray) -> bool:
    n, m = A.shape
    return A.sum() == n*m


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
