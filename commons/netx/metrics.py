__all__ = [
    "structural_hamming_distance",
    "structural_intervention_distance",
]

import networkx as nx
import numpy as np
from .graph import Graph
from .mat import adjacency_matrix
from .sidimpl.structIntervDist import struct_interv_dist


# ---------------------------------------------------------------------------
# structural_hamming_distance
# ---------------------------------------------------------------------------

def structural_hamming_distance_graph(G: Graph, H: Graph) -> float:
    assert G.order() == H.order(), "Incompatible graphs"
    Gam = adjacency_matrix(G)
    Ham = adjacency_matrix(H)
    return structural_hamming_distance_matrix(Gam, Ham)

def structural_hamming_distance_matrix(G: np.ndarray, H:np.ndarray) -> float:
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)
    n, m = G.shape
    shd = 0
    for i in range(n):
        for j in range(i + 1, m):
            if G[i, j] == H[i, j]:
                pass
            elif G[i, j] == H[j, i]:
                shd += 1
            else:
                shd += 1
    return shd

# ---------------------------------------------------------------------------
# structural_intervention_distance
# ---------------------------------------------------------------------------
# G = (V, E)
# PA(G,i)   parents  (1 step)
# CH(G,i)   children (1 step)
# DE(G, i)  descendants (recursive)
# AN(G, i)  ancestors   (recursive)
# ND(G, i)  non descendants = V \ DE(G,i)

# G(x, y, Z)
# In G, Z subset of V \ {x,y}
# 1) no z ∈ Z is a descendant of any w != x which lies on a directed path from x to y
# 2) and Z blocks all nondirected paths from x to y

# for P i paths(G, x, y):
#   for w in P
#       DEw = descendant(G, w)
#       |Z intersect DEw|

# def sid_step(G: Graph, u:NODE_TYPE, v:NODE_TYPE, Z:set[NODE_TYPE]) -> bool:
#     assert u not in Z and v not in Z
#
#     # check if there exists a path u->...->v
#     if not is_direct_connected(G, u, v):
#         # return True
#         pass
#
#     # no z ∈ Z is a descendant of any w != u which lies on a directed path from u to v
#     if not no_descendats(G, u, v, Z):
#         return False
#
#     #  Z blocks all nondirected paths from u to v
#     if not all_paths_blocked(G, u, v, Z):
#         return False
#     else:
#         return True
# # end
#
#
# def no_descendats(G: Graph, u: NODE_TYPE, v: NODE_TYPE, Z: Collection[NODE_TYPE]) -> bool:
#     # no z ∈ Z is a descendant of any w != u which lies on a directed path from u to v
#     # Note: SOME path, NOT ALL paths
#     assert u not in Z and v not in Z
#
#     for P in find_all_directed_paths(G, u, v):
#         if len(P) == 2: continue
#         for w in P:
#             if w == u: continue
#             DEw = descendants(G, w, recursive=True)
#             if len(DEw.intersection(Z)) > 0:
#                 return False
#     # end
#     return True
# # end
#
#
# def structural_intervention_distance(G: Graph, H: Graph) -> float:
#     # 2015 - Structural Intervention Distance (SID) for Evaluating Causal Graphs.pdf
#     assert G.order() == H.order(), f"Incompatible graphs: {G.order()}, {H.order()}"
#     N = list(G.nodes)
#
#     sid = 0
#     for i in N:
#         PAHi = netx.ancestors(H, i, recursive=False)    # ancestors    in H of i
#         DEGi = netx.descendants(G, i, recursive=True)   # descendants  in G of i
#         for j in N:
#             if i == j:
#                 continue
#             if j in PAHi and j in DEGi:
#                 sid += 1
#                 continue
#             if j in PAHi:
#                 continue
#
#             ss = sid_step(G, i, j, PAHi)
#             if not ss:
#                 sid += 1
#         # end
#     # end
#     return sid
# # end

def structural_intervention_distance_graph(G: Graph, H: Graph) -> float:
    Gam = nx.adjacency_matrix(G)
    Ham = nx.adjacency_matrix(H)
    return structural_intervention_distance_from_matrix(Gam, Ham)


def structural_intervention_distance_from_matrix(G: np.ndarray, H:np.ndarray) -> float:
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)
    # {'sid': 8, 'sidUpperBound': 8, 'sidLowerBound': 8, 'incorrectMat': array([...])}
    sid_res = struct_interv_dist(G, H)
    return sid_res["sid"]
# end

