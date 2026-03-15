
__all__ = [
    "is_partial_adjacency_matrix",
    "is_partial_directed_acyclic_graph",
    "enumerate_all_directed_graphs",
]

from typing import Iterator, Union
import networkx as nx
import netx
import numpy as np

from .types import EDGE_TYPE
from .dagfun import is_directed_acyclic_graph
from .graph import Graph

# a PDAG is a Graph where some edges have a direction and some are undirected
# It is possible to model a PDAG using a directed graph where the undirected
# edge is modelled as 2 directed edges:
#
#       u--v   ==   u->v & u<-v
#
# a PDAG can be converted in a set of DAGs

# ---------------------------------------------------------------------------
# is_partial_graph
# is_partial_adjacency_matrix
# ---------------------------------------------------------------------------

# def is_partial_graph(G: Union[nx.DiGraph, np.ndarray]) -> bool:
#     if isinstance(G, (nx.Graph, Graph)):
#         return is_partial_directed_acyclic_graph(G)
#     elif isinstance(G, np.ndarray):
#         return is_partial_adjacency_matrix(G)
#     else:
#         raise ValueError(f"Unsupported graph type: {type(G)}")


def is_partial_adjacency_matrix(A: np.ndarray) -> bool:
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] != 0 and A[j,i] != 0:
                return True
    return False


# ---------------------------------------------------------------------------
# is_partial_directed_acyclic_graph
# ---------------------------------------------------------------------------
# nx.is_directed_acyclic_graph(G)

def is_partial_directed_acyclic_graph(G: Graph) -> bool:
    """
    A 'partial directed graph' is a directed graph with edges {a->b} and {b->a}
    """
    if not nx.is_directed(G):
        return False

    edges = list(G.edges())
    for e in edges:
        u, v = e
        if (v, u) in edges:
            return True
    return False
# end


# ---------------------------------------------------------------------------
# enumerate_all_directed_graphs
# ---------------------------------------------------------------------------

def _directed_edges(G: Graph) -> list[EDGE_TYPE]:
    edges = G.edges()

    if not nx.is_directed(G):
        return list(edges)

    dedges = []
    for u, v in edges:
        if not G.has_edge(v, u):
            dedges.append((u, v))
    # end
    return dedges
# end


def _undirect_edges(G: Graph) -> list[EDGE_TYPE]:
    """
    Collect the list of undirected edges from a partial directed graph
    :param G: partial directed graph
    :return: list of undirected edges
    """
    edges = G.edges()

    uedges = []
    if not nx.is_directed(G):
        return list(edges)

    for u, v in edges:
        if G.has_edge(v, u):
            uedges.append((u, v))
    # end
    return uedges
# end

def _bitsgen(nbits: int) -> Iterator[list[int]]:
    bits = [0]*nbits
    ones = [1]*nbits
    while bits != ones:
        yield bits
        for i in range(nbits):
            if bits[i] == 0:
                bits[i] = 1
                break
            else:
                bits[i] = 0
    yield ones
# end


def enumerate_all_directed_graphs(G: nx.DiGraph, dag=False) -> Iterator[nx.DiGraph]:
    """
    Return the list of directed graphs from a partial directed graph
    :param G: partial directed graph
    :param dag: if to check for DAG
    :return:
    """
    dedges = _directed_edges(G)
    uedges = _undirect_edges(G)
    redges = [(e[1],e[0]) for e in uedges]
    nedges = len(uedges)

    for bits in _bitsgen(nedges):
        D = netx.create_like(G)
        D.add_edges_from(dedges)

        sedges = []
        for i in range(nedges):
            sedges.append(redges[i] if bits[i] else uedges[i])
        D.add_edges_from(sedges)

        if dag and not is_directed_acyclic_graph(D):
            continue
        yield D
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
