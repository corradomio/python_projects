
__all__ = [
    "is_partial_directed_acyclic_graph",
    "enumerate_directed_graphs",
    "partial_graph_undirected_edges",
]

import logging
from typing import Iterator

import networkx as nx

import netx
from .mat import _rand_int_iter, _bools
from .mat import ilog2
from .types import EDGE_TYPE


# a PDAG is a Graph where some edges have a direction and some are undirected
# It is possible to model a PDAG using a directed graph where the undirected
# edge is modelled as 2 directed edges:
#
#       u--v   ==   u->v & u<-v
#
# a PDAG can be converted in a set of DAGs


# ---------------------------------------------------------------------------
# is_partial_directed_acyclic_graph
# ---------------------------------------------------------------------------
# nx.is_directed_acyclic_graph(G)

def is_partial_directed_acyclic_graph(G: nx.DiGraph) -> bool:
    """
    A 'partial directed graph' is a directed graph with edges 'a->b' and 'b->a'
    """
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

# def _boolgen(nbits: int) -> Iterator[list[int]]:
#     """
#     Generate the list
#         [[0,0,0...],[1,0,0,...],[0,1,0,...],[1,1,0,...], ...]
#     (2^nbits elements)
#     :param nbits: number of bits to generate
#     :return: iterator on the list of bits
#     """
#     bits = [False]*nbits
#     ones = [True]*nbits
#     while bits != ones:
#         yield bits
#         for i in range(nbits):
#             if not bits[i]:
#                 bits[i] = True
#                 break
#             else:
#                 bits[i] = False
#     yield ones
# # end


def _g_select_edges(G: nx.DiGraph) -> tuple[list[EDGE_TYPE], list[EDGE_TYPE]]:
    assert nx.is_directed(G)
    edges = G.edges()

    dedges = set()
    uedges = set()
    for u, v in edges:
        if not G.has_edge(v, u):
            dedges.add((u, v))
        else:
            if u > v: u, v = v, u
            uedges.add((u, v))
    # end
    return list(dedges), list(uedges)


def enumerate_directed_graphs(G: nx.DiGraph, *, dag=False, max_count=256, max_tries=8192) -> Iterator[nx.DiGraph]:
    """
    Enumerate all direct graphs starting from the a partial directed graph G, that is
    a graph where there are undirected edges, edges present in both directions ('a->b' and 'a<-b').
    Because the number of directed graphs is 2^n( with n number of undirected edges) it could be useful
    to limit the number of graphs to generate

    :param G: partial directed graph
    :param dag: if to check for DAG
    :param max_count: max number of graphs to generate
    :return: iterator on generated graphs
    """
    assert isinstance(G, nx.DiGraph)
    # directed/undirected edges
    dedges, uedges = _g_select_edges(G)
    # reverse undirected edges
    redges = [(e[1],e[0]) for e in uedges]
    # number of undirected edges
    nedges = len(uedges)

    # there ae no undirected edges
    if nedges == 0:
        if nx.is_directed_acyclic_graph(G):
            yield G
        return

    # there are indirected edges
    max_edges = ilog2(max_count)
    if nedges <= max_edges:
        ints = range(2**nedges)
    else:
        logging.getLogger("netx").warning(f"PDAG contains {nedges} undirected edges exceeding the limit of {max_edges} edges")
        ulim = 2**nedges
        ints = _rand_int_iter(ulim)

    itry = 0
    icount = 0
    for bits in _bools(nedges, ints):
        if itry   > max_tries: break
        if icount > max_count: break

        D = netx.create_like(G)
        # add directed edges
        D.add_edges_from(dedges)
        # select a direction of undirected edge
        sedges = []
        for i in range(nedges):
            sedges.append(redges[i] if bits[i] else uedges[i])
        D.add_edges_from(sedges)
        # check for dag
        if dag and not nx.is_directed_acyclic_graph(D):
            itry += 1
            continue
        itry = 0
        icount += 1
        yield D
# end


def partial_graph_undirected_edges(G: nx.DiGraph) -> list[EDGE_TYPE]:
    dedges, uedges = _g_select_edges(G)
    return uedges
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
