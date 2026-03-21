#
# Generate graphs using adjacency matrices
#
__all__ = [
    "random_adjacency_matrix",
    "is_partial_adjacency_matrix",
    "connected_components_adjacency_matrix",
    "ancestors_adjacency_matrix",
    "descendants_adjacency_matrix",
    "enumerate_directed_adjacency_matrices",
    "undirected_edges_adjacency_matrix",
    "paths_adjacency_matrix",
]

from typing import Iterator, Generator
import logging
import numpy as np
from random import randrange
from .mat import ilog2, _rand_int_iter, _bools
from .types import EDGE_TYPE


def _where(pred) -> set[int]:
    return set(map(int, np.where(pred)[0]))

# ---------------------------------------------------------------------------
# connected_components_adjacency_matrix
# ---------------------------------------------------------------------------

def _neighbours_am(A: np.ndarray, u: int, undirected) -> set[int]:
    neigh = set()
    neigh.update(_where(A[u,:] == 1))
    if undirected:
        neigh.update(_where(A[:,u] == 1))
    return neigh


def connected_components_adjacency_matrix(A: np.ndarray) -> Iterator[set[int]]:
    n, _ = A.shape

    tovisit = set(range(n))
    while len(tovisit) > 0:
        u = tovisit.pop()
        visited = {u}
        neighbours = _neighbours_am(A, u, False)
        while len(neighbours) > 0:
            v = neighbours.pop()
            if v in visited: continue
            visited.add(v)
            neighbours.update(_neighbours_am(A, v, False))
        # end
        tovisit = tovisit.difference(visited)
        yield visited
    # end
# end


# ---------------------------------------------------------------------------
# ancestors_adjacency_matrix
# ancestors_adjacency_matrix
# ---------------------------------------------------------------------------

def ancestors_adjacency_matrix(A: np.ndarray, u: int, recursive=True) -> set[int]:
    if not recursive:
        return _where(A[:, u] == 1)

    visited = {u}
    tovisit = _where(A[:, u] == 1)
    while len(tovisit) > 0:
        s = tovisit.pop()
        if s in visited: continue
        visited.add(s)
        tovisit.update(_where(A[:, s]))

    return visited.difference([u])
# end


def descendants_adjacency_matrix(A: np.ndarray, u: int, recursive=True) -> set[int]:
    if not recursive:
        return _where(A[u, :] == 1)

    visited = {u}
    tovisit = _where(A[u, :] == 1)
    while len(tovisit) > 0:
        s = tovisit.pop()
        if s in visited: continue
        visited.add(s)
        tovisit.update(_where(A[s, :]))

    return visited.difference([u])
# end


# ---------------------------------------------------------------------------
# paths_adjacency_matrix
# ---------------------------------------------------------------------------

def _path_am(A, u_path: list[int], s: int, v: int, undirected=False) -> Iterator[list[int]]:
    if s == v:
        yield u_path
        return

    neigh = _neighbours_am(A, s, undirected).difference(u_path)
    for t in neigh:
        t_path = u_path + [t]
        yield from _path_am(A, t_path, t, v, undirected)


def paths_adjacency_matrix(A: np.ndarray, u: int, v: int, undirected=False) -> Iterator[list[int]]:
    n, _ = A.shape
    yield from _path_am(A, [u], u, v, undirected)
# end


# ---------------------------------------------------------------------------
# random_adjacency_matrix
# ---------------------------------------------------------------------------

def _random_am(n: int, k: int, loop: bool) -> np.ndarray:
    # adjacency matrix for random undirected graph
    assert k <= (n * n - 1) // 2
    A = np.zeros((n, n), np.int8)
    nedges = 0
    while nedges < k:
        u = randrange(n)
        v = randrange(n)
        if u == v and not loop or A[u, v] == 1: continue
        A[u, v] = 1
        A[v, u] = 1
        nedges += 1
    return A


def _random_directed_am(n: int, k: int) -> np.ndarray:
    # adjacency matrix for random directed graph
    assert k <= (n * n - 1)
    A = np.zeros((n, n), np.int8)
    nedges = 0
    while nedges < k:
        u = randrange(n)
        v = randrange(n)
        if u == v or A[u, v] == 1 or A[v, u] == 1: continue
        A[u, v] = 1
        nedges += 1
    return A


def _random_dag_am(n: int, k: int) -> np.ndarray:
    # adjacency matrix for random directed acyclic graph
    assert k <= (n * n - 1)
    A = np.zeros((n, n), np.int8)
    nedges = 0
    while nedges < k:
        u = randrange(n)
        v = randrange(n)
        if u > v: u, v = v, u
        if u == v or A[u, v] == 1: continue
        A[u, v] = 1
        nedges += 1
    return A


def random_adjacency_matrix(
        n: int, k: int,
        directed: bool = False,
        acyclic: bool = False,
        connected: bool= False,
        loop: bool = False
) -> np.ndarray:
    """

    :param n: n of nodes
    :param k: n of edges
    :param directed: if the graph must be directed
    :param acyclic: if the graph must be directed and acyclic
    :param loop: if the loops are permitted
    :param connected: if the graph must be connected
    :return:
    """
    if not directed:
        return _random_am(n, k, loop)
    if not acyclic:
        return _random_directed_am(n, k)
    else:
        return _random_dag_am(n, k)
# end


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
# enumerate_directed_adjacency_matrices
# ---------------------------------------------------------------------------

def _undirected_edges_am(A: np.ndarray) -> list[EDGE_TYPE]:
    n, _ = A.shape
    uedges = set()

    for u in range(n):
        for v in range(u+1, n):
            if A[u,v] == 1 and A[v, u] == 1:
                uedges.add((u, v))
    return list(uedges)


def _is_dag_dfs(A: np.ndarray) -> bool:
    # deep first search
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
    uedges = _undirected_edges_am(A)
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



# ---------------------------------------------------------------------------
# undirected_edges_adjacency_matrix
# ---------------------------------------------------------------------------

def undirected_edges_adjacency_matrix(A: np.ndarray) -> list[EDGE_TYPE]:
    return _undirected_edges_am(A)
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
