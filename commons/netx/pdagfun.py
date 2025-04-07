from typing import Iterator
import networkx as nx
from .dagfun import is_directed_acyclic_graph


# ---------------------------------------------------------------------------
# is_partial_directed
# ---------------------------------------------------------------------------

def is_partial_directed(G) -> bool:
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


def undirect_edges(G) -> list[tuple[int, int]]:
    """
    Collect the list of undirected edges from a partial directed graph
    :param G: partial directed graph
    :return: list of undirected edges
    """
    edges = list(G.edges())

    uedges = []
    if not nx.is_directed(G):
        return edges

    n = len(edges)
    for i in range(n):
        e = edges[i]
        u, v = e
        f = (v, u)
        for j in range(i+1, n):
            if f == edges[j]:
                uedges.append(e)
                break
    # end
    return uedges
# end


# ---------------------------------------------------------------------------
# pdag_enum
# ---------------------------------------------------------------------------

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


def pdag_enum(G: nx.Graph, dag=False) -> Iterator[nx.Graph]:
    """
    Return the list of directed graphs from a partial directed graph
    :param G: partial directed graph
    :param dag: if to check for DAG
    :return:
    """
    uedges = undirect_edges(G)
    redges = [(e[1],e[0]) for e in uedges]
    nedges = len(uedges)

    for bits in _bitsgen(nedges):
        D = G.copy()
        for i in range(nedges):
            if bits[i]: # 1
                D.remove_edge(*redges[i])
            else:
                D.remove_edge(*uedges[i])
        if dag and not is_directed_acyclic_graph(D):
            continue
        yield D
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
