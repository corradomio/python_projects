import numpy as np
from .graph import Graph
from scipy.sparse import csr_array


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------
# direct=False,
# loops=False,
# multi=False,
# acyclic=False,

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

    if create_using is None:
        G = Graph(direct=is_direct(), loops=has_loops(), multi=False, acyclic=False)
    else:
        G = create_using()

    for v in range(n):
        G.add_node(v)

    for u in range(n):
        for v in range(n):
            if adjacency_matrix[u,v] == 1:
                G.add_edge(u, v)

    return G
# end


def from_numpy_matrix(adjacency_matrix: np.ndarray) -> Graph:
    return from_adjacency_matrix(adjacency_matrix)
# end


def adjacency_matrix(G: Graph) -> np.ndarray:
    n = G.order()
    is_directed = G.is_directed()
    A = np.zeros((n, n), dtype=int)
    for e in G.edges():
        u, v = e
        w = 1
        if is_directed:
            A[u, v] = w
        else:
            A[u, v] = w
            A[v, u] = w
    # end
    return A
# end
