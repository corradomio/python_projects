from typing import Collection
from stdlib.is_instance import is_instance
from .dagfun import descendants, ancestors
from .graph import Graph, NODE_TYPE
from .mat import adjacency_matrix
from .paths import find_all_undirected_paths, find_all_directed_paths


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

    for k in range(1,n-1):
        a = P[k-1]
        u = P[k]
        b = P[k+1]

        if G.has_edge(a, u) and G.has_edge(u, b):
            chains.add(u)
        if G.has_edge(b, u) and G.edges(u, a):
            chains.add(u)

    return chains
# end


def path_forks(G: Graph, P: Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    assert isinstance(P, list)
    n = len(P)

    # A<-u->B
    forks = set()

    for k in range(1,n-1):
        a = P[k-1]
        u = P[k]
        b = P[k+1]

        if G.has_edge(u, a) and G.has_edge(u, b):
            forks.add(u)

    return forks
# end


def path_colliders(G: Graph, P: Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    assert isinstance(P, list)
    n = len(P)

    # A->u<-B
    colliders = set()

    for k in range(1,n-1):
        a = P[k-1]
        u = P[k]
        b = P[k+1]

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
# We first define the "d"-separation of a trail then we will define the "d"-separation of two nodes in terms of that.
# Let P be a trail from node u to v. A trail is a loop-free, undirected (i.e. all edge directions are ignored) path
# between two nodes.
# Then P is said to be d-separated by a set of nodes Z if any of the following conditions holds:
# 1) P contains a chain,    u ⋯ ← m ← ⋯ v, such that the middle node m is in Z,
# 2) P contains a fork,     u ⋯ ← m → ⋯ v, such that the middle node m is in Z,
# 3) P contains a collider, u ⋯ → m ← ⋯ v, such that the middle node m is not in Z and no descendant of m is in Z.
#

def is_path_blocked(G: Graph, P: list[NODE_TYPE], Z: Collection[NODE_TYPE]) -> bool:
    # check for P=[u,v] NECESSARY
    if len(P) <= 2:
        return True

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
    return blocked
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

def is_d_separated(G, u: NODE_TYPE, v: NODE_TYPE, Z: Collection[NODE_TYPE]) -> bool:
    U = [u] if is_instance(u, NODE_TYPE) else u
    V = [v] if is_instance(v, NODE_TYPE) else v
    Z = [Z] if is_instance(Z, NODE_TYPE) else Z

    for u in U:
        for v in V:
            # if not all_paths_blocked(G, u, v, Z):
            #     return False
            for P in find_all_undirected_paths(G, u, v):
                if not is_path_blocked(G, P, Z):
                    return False
    return True
# end


# ---------------------------------------------------------------------------
# are_markov_equivalents
# ---------------------------------------------------------------------------

def _find_colliders(G: Graph) -> set[NODE_TYPE]:
    colliders = set()
    for n in G.nodes():
        # if a node has multiple inputs and 1+ outputs,
        # it can be part of a chain
        if G.in_degree(n) >= 2 and G.out_degree(n) == 0:
            colliders.add(n)
    return colliders


def is_markov_equivalent(G: Graph, H: Graph) ->bool:
    """
    The graphs G and H are Markov-equivalent if they have the same
    skeleton and the same set of colliders.
    The same skeleton means that the nodes u and v are connected as
    'u->v' OR 'u<-v'.
    A collider is a node with an input degree >= 2

    :param G:
    :param H:
    :return:
    """
    assert G.is_directed(), "Graph G must be directed"
    assert H.is_directed(), "Graph H must be directed"

    # 1) che if they have the same skeleton, that is the have the
    #    same undirected edges (the direction is not important)
    if G.number_of_nodes() != H.number_of_nodes():
        return False
    if G.number_of_edges() != H.number_of_edges():
        return False
    for u, v in G.edges:
        if not H.has_edge(u, v) and not H.has_edge(v, u):
            return False

    # 2) check if both graphs have the same set of colliders
    if _find_colliders(G) != _find_colliders(H):
        return False
    return True
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
