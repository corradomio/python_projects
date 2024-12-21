from collections import deque
from typing import Iterator

import networkx as nx

from .graph import Graph, NODE_TYPE


# ---------------------------------------------------------------------------
# is_dag
# ---------------------------------------------------------------------------

def is_dag(G: Graph) -> bool:
    if not nx.is_directed(G):
        return False
    try:
        nx.find_cycle(G, orientation='original')
        return False
    except nx.NetworkXNoCycle:
        return True
# end


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

#
# G.cache: dict[str, dict[...]]
#


def sources(G: Graph) -> set[NODE_TYPE]:
    """
    Direct Graph sources: nodes with 0 in-degree
    """
    assert G.is_directed()

    if "nodes" not in G.cache["sources"]:
        slist = []
        for n in G.nodes_:
            if G.in_degree(n) == 0:
                slist.append(n)

        G.cache["sources"]["nodes"] = set(slist)
    return G.cache["sources"]["nodes"]


def destinations(G: Graph) -> set[NODE_TYPE]:
    """
    Direct Graph destinations: nodes with 0 out-degree
    """
    assert G.is_directed()
    if "nodes" not in G.cache["destinations"]:
        dlist = []
        for n in G.nodes_:
            if G.out_degree(n) == 0:
                dlist.append(n)
        G.cache["destinations"]["nodes"] = set(dlist)
    return G.cache["destinations"]["nodes"]


def parents(G: Graph, n: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Parents of the current node
    """
    assert G.is_directed()
    return set(G.pred[n])


def children(G: Graph, n: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Children of the current node
    """
    assert G.is_directed()
    return set(G.succ[n])


def ancestors(G: Graph, v: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Ancestors of the current node
    """
    assert G.is_directed()
    if v not in G.cache["ancestors"]:
        waiting = deque()
        ancestors: set[int] = set()
        waiting.extend(parents(G, v))
        while waiting:
            u = waiting.popleft()
            if u not in ancestors:
                ancestors.add(u)
                waiting.extend(G.pred[u])
        G.cache["ancestors"][v] = ancestors
    # end
    return G.cache["ancestors"][v]


def descendants(G: Graph, u: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Descendants of the current node
    """
    assert G.is_directed()
    if u not in G.cache["descendants"]:
        waiting = deque()
        descendants: set[int] = set()
        waiting.extend(children(G, u))
        while waiting:
            v = waiting.popleft()
            if v not in descendants:
                descendants.add(v)
                waiting.extend(G.succ[v])
        G.cache["descendants"][u] = descendants
    # end
    return G.cache["descendants"][u]


# ---------------------------------------------------------------------------
# find_paths
# find_all_paths
# ---------------------------------------------------------------------------

def _find_path(G: Graph, u_path: list[NODE_TYPE], s: int, v: NODE_TYPE, u_processed: set[NODE_TYPE]) \
        -> Iterator[list[NODE_TYPE]]:
    if s == v:
        yield u_path + [v]

    if s in u_processed:
        return

    for t in G.succ[s]:
        yield from _find_path(G, u_path + [s], t, v, u_processed | {s})


def find_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Find all paths between the specified nodes. It returns an iterator
    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: an iterator on uv paths
    """
    assert G.is_directed()

    u_processed = {u}
    for t in G.succ[u]:
        yield from _find_path(G, [u], t, v, u_processed)
# end


def find_all_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> list[list[NODE_TYPE]]:
    """
    Find all paths between the specified nodes. It returns the list of paths,
    Note: the list of paths is saved the graph 'cache'
    Note: the implementation is based on 'find_paths(G, u, v)'

    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: the list of paths
    """
    uv = u, v
    if uv not in G.cache["find_paths"]:
        uv_paths = list(find_paths(G, u, v))
        G.cache["find_paths"][uv] = uv_paths
    return G.cache["find_paths"][uv]
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
