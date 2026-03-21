__all__ = [
    "structural_hamming_distance",
    "structural_intervention_distance",
]

import networkx as nx
import numpy as np

from .graph_am import enumerate_directed_adjacency_matrices
from .pdagfun import enumerate_directed_graphs
from .mat import adjacency_matrix
from .sidimpl.structIntervDistDAG import structIntervDist
from .sidimpl.hammingDist import hammingDist


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _mean(l) -> float:
    if len(l) == 0:
        return -1
    return sum(l)/len(l)


# ---------------------------------------------------------------------------
# hamming_distance
# ---------------------------------------------------------------------------

def hamming_distance(G: nx.DiGraph, H: nx.DiGraph, all_mistakes_one: bool=True) -> float:
    """
    Hamming distance based on R implementation converted in Python
    :param G:
    :param H:
    :param all_mistakes_one:
    :return:
    """
    if isinstance(G, (nx.DiGraph)):
        assert G.order() == H.order(), "Incompatible graphs"
        Gam = adjacency_matrix(G)
        Ham = adjacency_matrix(H)
    else:
        Gam = G
        Ham = H

    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)
    return hammingDist(Gam, Ham, all_mistakes_one=all_mistakes_one)
# end


# ---------------------------------------------------------------------------
# structural_hamming_distance
# ---------------------------------------------------------------------------

def _structural_hamming_distance(G: np.ndarray, H: np.ndarray, all_mistakes_one: bool=True) -> float:
    diff = np.abs(G - H)
    if all_mistakes_one:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1
        shd = np.sum(diff) / 2
    else:
        shd = np.sum(diff)
    return shd
# end


def structural_hamming_distance(G: np.ndarray, H: np.ndarray, all_mistakes_one: bool=True) -> float:
    """
    Hamming distance between two DAG
    :param G: ground truth DAG
    :param H: DAG
    :param all_mistakes_one:
    :return:
    """
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)
    return _structural_hamming_distance(G, H, all_mistakes_one=all_mistakes_one)
# end


def structural_hamming_distance_pdag(G: np.ndarray, H: np.ndarray, all_mistakes_one: bool=True, *,
        max_count=256, max_tries=8192) -> float:
    """
    Hamming distance between a DAG and a PDAG
    :param G: ground truth DAG
    :param H: PDAG
    :param all_mistakes_one:
    :return:
    """
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)

    shd_list = []
    for Gi in enumerate_directed_adjacency_matrices(G, dag=True, max_count=max_count, max_tries=max_tries):
        for Hj in enumerate_directed_adjacency_matrices(H, dag=True, max_count=max_count, max_tries=max_tries):
            shd = _structural_hamming_distance(Gi, Hj, all_mistakes_one=all_mistakes_one)
            shd_list.append(shd)
    return _mean(shd_list)
# end


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

def structural_intervention_distance(G: np.ndarray, H: np.ndarray) -> float:
    """
    Compute the Structural Intervention Distance between two DAGs,
    Based on the R implementation converted in Python.

    :param G: DAG ground truth
    :param H: causal graph
    :return:
    """
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)
    assert G.shape == H.shape
    # sid_res: {'sid': 8, 'sidUpperBound': 8, 'sidLowerBound': 8, 'incorrectMat': array([...])}

    return float(structIntervDist(G, H)["sid"])
# end


def structural_intervention_distance_pdag(G: np.ndarray, H: np.ndarray, *,
        max_count=256, max_tries=8192) -> float:
    """
    Hamming distance between a DAG and a PDAG
    :param G: ground truth DAG
    :param H: PDAG
    :param all_mistakes_one:
    :return:
    """
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)

    sid_list = []
    for Gi in enumerate_directed_graphs(G, dag=True, max_count=max_count, max_tries=max_tries):
        for Hj in enumerate_directed_adjacency_matrices(H, dag=True, max_count=max_count, max_tries=max_tries):
            sid = float(structIntervDist(Gi, Hj)["sid"])
            sid_list.append(sid)
    return _mean(sid_list)
# end


# ---------------------------------------------------------------------------
# symmetric_intervention_distance
# ---------------------------------------------------------------------------

def symmetric_intervention_distance(G: np.ndarray, H: np.ndarray):
    """
        Compute the Structural Intervention Distance between two DAGs,
        Based on the R implementation converted in Python.

        :param G: DAG ground truth
        :param H: causal graph
        :return:
        """
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)
    assert G.shape == H.shape
    # sid_res: {'sid': 8, 'sidUpperBound': 8, 'sidLowerBound': 8, 'incorrectMat': array([...])}

    sidgh = structIntervDist(G, H)["sid"]
    sidhg = structIntervDist(H, G)["sid"]
    return (sidgh + sidhg)/2
# end


def symmetric_intervention_distance_pdag(G: np.ndarray, H: np.ndarray, *,
        max_count=256, max_tries=8192) -> float:
    """
    Hamming distance between a DAG and a PDAG
    :param G: ground truth DAG
    :param H: PDAG
    :param all_mistakes_one:
    :return:
    """
    assert isinstance(G, np.ndarray)
    assert isinstance(H, np.ndarray)

    ssid_list = []
    for Gi in enumerate_directed_adjacency_matrices(G, dag=True, max_count=max_count, max_tries=max_tries):
        for Hj in enumerate_directed_adjacency_matrices(H, dag=True, max_count=max_count, max_tries=max_tries):
            sidgh = float(structIntervDist(Gi, Hj)["sid"])
            sidhg = float(structIntervDist(Hj, Gi)["sid"])
            ssid_list += [sidgh, sidhg]
    return _mean(ssid_list)
# end


# ---------------------------------------------------------------------------
# d_separation_distance
# ---------------------------------------------------------------------------

def _d_separation_matrix(G: nx.DiGraph) -> np.ndarray:
    n = G.order()
    dsep = np.ndarray((n,n), dtype=np.int8)
    for u in G.nodes:
        for v in G.nodes:
            if u == v: continue
            dsep[u,v] = nx.is_d_separator(G, u, v, set())
    return dsep
# end


def d_separation_distance(G: nx.DiGraph, H: nx.DiGraph) -> float:
    assert isinstance(G, nx.DiGraph)
    assert isinstance(H, nx.DiGraph)
    assert G.order() == H.order()

    gdsep = _d_separation_matrix(G)
    hdsep = _d_separation_matrix(H)
    # hamming distance
    return float((np.abs(gdsep - hdsep)).sum())
# end


def d_separation_distance_pdag(G: nx.DiGraph, H: nx.DiGraph, *,
        max_count=256, max_tries=8192) -> float:
    assert isinstance(G, nx.DiGraph)
    assert isinstance(H, nx.DiGraph)
    assert G.order() == H.order()

    dsd_list = []
    for Gi in enumerate_directed_graphs(G, dag=True, max_count=max_count, max_tries=max_tries):
        gdsep = _d_separation_matrix(Gi)
        for Hj in enumerate_directed_graphs(H, dag=True, max_count=max_count, max_tries=max_tries):
            hdsep = _d_separation_matrix(Hj)
            dsd = float((np.abs(gdsep - hdsep)).sum())
            dsd_list.append(dsd)
    return _mean(dsd_list)
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
