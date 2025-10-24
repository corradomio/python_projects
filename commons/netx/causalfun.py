from typing import Collection, Optional
from stdlib.is_instance import is_instance
from .graph import Graph, NODE_TYPE
from .paths import find_all_undirected_paths, find_all_directed_paths
from .dagfun import descendants

# ---------------------------------------------------------------------------
# chain, fork, collider
# ---------------------------------------------------------------------------
# chain         A->X->B | A<-X<-B
# fork          A<-X->B             cofounder
# colliders     A->X<-B             join???


def path_chains(G: Graph, P: Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    assert isinstance(P, list)
    n = len(P)

    # A->u->B
    # A<-u<-B
    chains = set()
    for k in range(1, n-1):
        u = P[k]
        a = P[k - 1]
        b = P[k + 1]

        if G.has_edge(a, u) and G.has_edge(u, b):
            chains.add(u)
        if G.has_edge(b, u) and G.has_edge(u, a):
            chains.add(u)
    return chains
# end


def path_forks(G: Graph, P: Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    assert isinstance(P, list)
    n = len(P)

    # A<-u->B
    forks = set()
    for k in range(1, n - 1):
        u = P[k]
        a = P[k - 1]
        b = P[k + 1]

        if G.has_edge(u, a) and G.has_edge(u, b):
            forks.add(u)
    return forks
# end


def path_colliders(G: Graph, P: Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    assert isinstance(P, list)
    n = len(P)

    # A->u<-B
    colliders = set()
    for k in range(1, n-1):
        u = P[k]
        a = P[k - 1]
        b = P[k + 1]

        if G.has_edge(a, u) and G.has_edge(b, u):
            colliders.add(u)
    return colliders
# end


#
# In causal inference, a path is blocked by a set of nodes (Z)
# if it contains a chain or fork where the middle node is in Z,
# or a collider where the middle node is not in Z or any of its descendants.
#
# When all paths between two variables are blocked by Z,
# the variables are "d-separated" and are conditionally independent given Z.
# This concept, known as d-separation, is crucial for determining
# which variables to control for to establish a causal relationship.
#
# https://fiveable.me/causal-inference/unit-9/d-separation-backdoor-criterion/study-guide/JmKovLOknVvGzRBz
#
# https://en.wikipedia.org/wiki/Bayesian_network#d-separation
# We first define the "d"-separation of a trail and then we will define the "d"-separation of two nodes in terms of that.
# Let P be a trail from node u to v. A trail is a loop-free, undirected (i.e. all edge directions are ignored) path
# between two nodes.
# Then P is said to be d-separated by a set of nodes Z if any of the following conditions holds:
# 1) P contains a chain,    u ⋯ ← m ← ⋯ v, such that the middle node m is in Z,
# 2) P contains a fork,     u ⋯ ← m → ⋯ v, such that the middle node m is in Z,
# 3) P contains a collider, u ⋯ → m ← ⋯ v, such that the middle node m is not in Z and no descendant of m is in Z.
#

def is_path_blocked(G: Graph, P: list[NODE_TYPE], Z: Collection[NODE_TYPE]) -> bool:
    assert isinstance(G, Graph), f"Unsupported graphs of type {type(G)}"

    # check for P=[u,v] not necessary

    # chain where the middle node m is in Z
    J = path_chains(G, P)
    if len(J.intersection(Z)) > 0:
        return True

    # fork where the middle node is in Z
    F = path_forks(G, P)
    if len(F.intersection(Z)) > 0:
        return True

    # collider where the middle node m is not in Z and no descendant of m is in Z.
    # Note: it is enough to find ONE collider satisfying the conditions
    C = path_colliders(G, P)
    blocked = False
    for c in C:
        D = descendants(G, c, recursive=True)
        if c not in Z and len(D.intersection(Z)) == 0:
            blocked = True
            break
    if blocked:
        return True

    # if len(P) == 2 and len(Z) == 0:
    #     return True

    else:
        return False
# end


def all_paths_blocked(G: Graph, u: NODE_TYPE, v: NODE_TYPE, Z: Collection[NODE_TYPE]) -> bool:
    assert u not in Z and v not in Z

    for P in find_all_undirected_paths(G, u, v):
        if not is_path_blocked(G, P, Z):
            return False
    # end
    return True
# end


# d)rectional-separation

def is_d_separator(G: Graph, U: Collection[NODE_TYPE], V: Collection[NODE_TYPE], Z: Collection[NODE_TYPE]) -> bool:
    if is_instance(U, NODE_TYPE): U = [U]
    if is_instance(V, NODE_TYPE): V = [V]
    if is_instance(Z, NODE_TYPE): Z = [Z]

    for u in U:
        for v in V:
            if not all_paths_blocked(G, u, v, Z):
                return False
    return True
# end


def no_descendats(G: Graph, u: NODE_TYPE, v: NODE_TYPE, Z: Collection[NODE_TYPE]) -> bool:
    # no z ∈ Z is a descendant of any w != u which lies on a directed path from u to v
    # Note: SOME path, NOT ALL paths
    assert u not in Z and v not in Z

    for P in find_all_directed_paths(G, u, v):
        if len(P) == 2: continue
        for w in P:
            if w == u: continue
            DEw = descendants(G, w, recursive=True)
            if len(DEw.intersection(Z)) > 0:
                return False
    # end
    return True
# end


# ---------------------------------------------------------------------------
# structural_hamming_distance
# ---------------------------------------------------------------------------

def structural_hamming_distance(G: Graph, H: Graph) -> float:
    assert G.order() == H.order(), "Incompatible graphs"
    Gam = adjacency_matrix(G)
    Ham = adjacency_matrix(H)
    n, m = Gam.shape
    # shd = np.abs(G-H).sum()
    shd = 0
    for i in range(n):
        for j in range(i + 1, m):
            if Gam[i, j] == Ham[i, j]:
                pass
            elif Gam[i, j] == Ham[j, i]:
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

def sid_step(G: Graph, u:NODE_TYPE, v:NODE_TYPE, Z:set[NODE_TYPE]) -> bool:
    assert u not in Z and v not in Z

    # no z ∈ Z is a descendant of any w != u which lies on a directed path from u to v
    if not no_descendats(G, u, v, Z):
        return False
    #  Z blocks all nondirected paths from u to v
    if not all_paths_blocked(G, u, v, Z):
        return False
    else:
        return True
# end


def structural_intervention_distance(G: Graph, H: Graph) -> float:
    # 2015 - Structural Intervention Distance (SID) for Evaluating Causal Graphs.pdf
    assert G.order() == H.order(), f"Incompatible graphs: {G.order()}, {H.order()}"
    N = list(G.nodes)

    sid = 0
    for i in N:
        PAHi = netx.ancestors(H, i, recursive=True)    # ancestors    in H of i
        DEGi = netx.descendants(G, i, recursive=True)   # descendants  in G of i
        for j in N:
            if i == j:
                continue
            if j in PAHi and j in DEGi:
                sid += 1
                continue
            if j in PAHi:
                continue

            ss = sid_step(G, i, j, PAHi)
            if not ss:
                sid += 1
        # end
    # end
    return sid
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
