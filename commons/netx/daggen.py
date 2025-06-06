import random as rnd
from typing import Generator

import networkx as nx
import numpy as np

from stdlib.iset import ilexsubset, imembers


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
# random_dag
# extends_dag
# ---------------------------------------------------------------------------

def random_dag(n: int, m: int, connected=True, create_using=None):
    """
    Generate a random directed acyclic graph

    :param n: n of nodes
    :param m: minimum n of edges
    :param connected: if the DAG must be connected
    :returns: the generated Directed Acyclic Graph
    """
    assert m <= (n*(n-1))/2, "Too edges specified"
    # assert connected and m >= (n-1), "Not enough edges for a connected graph"

    G = nx.DiGraph() if create_using is None else create_using()
    G.add_nodes_from(list(range(n)))
    return extends_dag(G, m, connected=connected)
# end


def extends_dag(G: nx.DiGraph, m: int, connected=True):
    """
    Extends the DAG to have 'm' edges.

    :param G: DAG to extend
    :param m: minimum n of edges
    :param connected: if the DAG must be connected
    :param seed: if specified, used to initialize the random number generator
    :returns: the original Directed Acyclic Graph with extra edges
    """
    n = G.order()

    completed = m <= len(G.edges())
    while not completed:
        u = rnd.randrange(n)
        v = rnd.randrange(n)
        # exclude the loops
        if u == v: continue
        # keep u->v only if u < v
        if u > v: u, v = v, u
        # skip already existent edges,
        if G.has_edge(u, v): continue

        # add the edge. NO cycles are created for construction
        G.add_edge(u, v)

        # check is the DAG must be connected OR it is reached the
        # required number of edges
        completed = not (connected and not nx.is_weakly_connected(G) or len(G.edges()) < m)
    # end
    return G
# end


# ---------------------------------------------------------------------------
# dag_enum
# ---------------------------------------------------------------------------
# DAG: represented by a upper triangular matrix
# Constraints: all rows and all columns must be not 0

def iset_to_amdag(S: int, n: int) -> np.ndarray:
    """
    Set to adjacency matrix for  DAG
    :param S: set as integer
    :param n: n of nodes
    :return: adjacency matrix
    """

    def i2rc(i: int) -> tuple[int, int]:
        """vector's index to matrix's row/col"""
        r, c = 0, 1
        m = n - 1
        while i >= m:
            r += 1
            i -= m
            m -= 1
            c += 1
        c += i
        return r, c

    A = np.zeros((n, n), dtype=np.int8)
    for i in imembers(S):
        r, c = i2rc(i)
        A[r, c] = 1
    return A
# end


def dag_enum(n: int, create_using=None) -> Generator:
    """
    Generate all connected DAGs with n nodes
    :param n: n of nodes
    :return: an iterator on the generated DAGs
    """
    N = n * (n - 1) // 2

    if create_using is None:
        create_using = nx.DiGraph

    for S in ilexsubset(n=N, k=(n - 1, N)):
        A = iset_to_amdag(S, n)

        G = nx.from_numpy_array(A, create_using=create_using)
        if not nx.is_weakly_connected(G):
            continue
        yield G
# end


def from_numpy_array(A: np.ndarray, create_using=None):
    """
    Create a Graph from a numpy adjacency matrix

    :param A: adjacency matrix
    :param create_using: a already graph to fill
        otherwise it is created a Graph if the matrix is simmetryc
        else a DiGraph
    :return: the graph satisfying the adjacency matrix
    """
    symmetric = False if create_using else is_symmetric(A)
    if create_using:
        G = create_using()
    elif symmetric:
        G = nx.Graph()
    else:
        G = nx.DiGraph()

    n = A.shape[0]

    G.add_nodes_from(range(n))

    for u in range(n):
        for v in range(n):
            if A[u, v]:
                G.add_edge(u, v)

    return G
# end
