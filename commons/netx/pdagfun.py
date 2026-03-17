
__all__ = [
    "is_partial_adjacency_matrix",
    "is_partial_directed_acyclic_graph",
    "enumerate_directed_graphs",
    "enumerate_directed_adjacency_matrices",
    "partial_graph_undirected_edges",
    "adjacency_matrix_undirected_edges"
]

from typing import Iterator, Union
import logging
import random
import networkx as nx
import netx
import numpy as np

from .types import EDGE_TYPE, uedge
from .dagfun import is_directed_acyclic_graph
from .graph import Graph
from .mat import ilog2

# a PDAG is a Graph where some edges have a direction and some are undirected
# It is possible to model a PDAG using a directed graph where the undirected
# edge is modelled as 2 directed edges:
#
#       u--v   ==   u->v & u<-v
#
# a PDAG can be converted in a set of DAGs

# ---------------------------------------------------------------------------
# is_partial_adjacency_matrix
# ---------------------------------------------------------------------------

def is_partial_adjacency_matrix(A: np.ndarray) -> bool:
    """
    A 'partial adjacency matrix' is a matrix with edges 'a->b' and 'b->a'
    :param A:
    :return:
    """
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            if A[i,j] != 0 and A[j,i] != 0:
                return True
    return False
# end


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

def _bools(nbits: int, iseq: Iterator[int]):
    for ival in iseq:
        bits = [False]*nbits
        m = 1
        for j in range(nbits):
            if ival & m:
                bits[j] = True
            m <<= 1
        yield bits


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


# ---------------------------------------------------------------------------
# enumerate_all_directed_adjacency_matrices
# ---------------------------------------------------------------------------

def _rand_int_iter(ulim: int) -> Iterator[int]:
    while True:
        yield random.randrange(ulim)
# end


def _a_undirected_edges(A: np.ndarray) -> list[EDGE_TYPE]:
    n, _ = A.shape
    uedges = set()

    for u in range(n):
        for v in range(u+1, n):
            if A[u,v] == 1 and A[v, u] == 1:
                uedges.add((u, v))
    return list(uedges)
# end


def _is_dag_dfs(A: np.ndarray) -> bool:
    num_nodes = len(A)
    # 0: unvisited, 1: visiting (grey), 2: fully visited (black)
    visited_status = [0] * num_nodes

    def dfs_check_cycle(node):
        visited_status[node] = 1 # Mark as grey (visiting)

        for neighbor in range(num_nodes):
            if A[node, neighbor] == 1: # Check for an edge
                if visited_status[neighbor] == 1:
                    return True  # Cycle found (back edge to a grey node)
                if visited_status[neighbor] == 0:
                    if dfs_check_cycle(neighbor):
                        return True

        visited_status[node] = 2 # Mark as black (fully visited)
        return False

    # Iterate over all nodes to cover disconnected components
    for i in range(num_nodes):
        if visited_status[i] == 0:
            if dfs_check_cycle(i):
                return False # Not a DAG (cycle found)

    return True # It is a DAG
# end


def enumerate_directed_adjacency_matrices(A: np.ndarray, *, dag=False, max_count=256, max_tries=8192) -> Iterator[np.ndarray]:
    """
    Enumerate all direct graphs starting from the adjacency matrix of a partial directed graph, that is
    a graph where some edges are undirected.

    :param A: adjacency matrix
    :param dag: if to check for DAG
    :param max_count: max graphs to generate
    :param max_tries: max number of graphs to check for DAG
    :return: iterator on generated directed graphd
    """
    uedges = _a_undirected_edges(A)
    nedges = len(uedges)

    if nedges == 0:
        if _is_dag_dfs(A):
            yield A
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

        D = A.copy()
        for i in range(nedges):
            u, v = uedges[i]
            if bits[i]:
                D[u, v] = 0
            else:
                D[v, u] = 0
        # check for dag
        if dag and not _is_dag_dfs(D):
            itry += 1
            continue
        itry = 0
        icount += 1
        yield D
# end


def adjacency_matrix_undirected_edges(A: np.ndarray) -> list[EDGE_TYPE]:
    return _a_undirected_edges(A)


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
