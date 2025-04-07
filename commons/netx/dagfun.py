from collections import deque
from typing import Iterator, Collection

import networkx as nx

from stdlib import is_instance
from .gclass import is_networkx_graph
from .graph import Graph, NODE_TYPE
from .cache import check_cache


#
#   is_directed_acyclic_graph
#   sources
#   destinations
#
#   predecessors == parents   (single step)
#   succesors == children     (single step)
#
#   ancestors   (recursive)
#   descendants (recursive)
#
#   find_paths
#   find_all_paths
#


# ---------------------------------------------------------------------------
# is_directed_acyclic_graph
# ---------------------------------------------------------------------------

def is_directed_acyclic_graph(G: Graph) -> bool:
    """
    Check if the graph is a DAG (Directed Acyclic Graph).
    If it is not directed, it is not a DAG
    If there are cycles, it is not a DAG

    :param G: graph to check
    :return: true or false
    """
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
#   sources
#   destinations
#   predecessors == parents
#   successors == children
#   ancestors
#   descendante

#
# G.cache: dict[str, dict[...]]
#

def sources(G: Graph) -> set[NODE_TYPE]:
    """
    Direct Graph sources: nodes with 0 in-degree
    """
    assert G.is_directed()

    check_cache(G)

    if "nodes" not in G.cache["sources"]:
        slist = []
        for n in G.nodes():
            if G.in_degree(n) == 0:
                slist.append(n)

        G.cache["sources"]["nodes"] = set(slist)
    return G.cache["sources"]["nodes"]
# end


def destinations(G: Graph) -> set[NODE_TYPE]:
    """
    Direct Graph destinations: nodes with 0 out-degree
    """
    assert G.is_directed()

    check_cache(G)

    if "nodes" not in G.cache["destinations"]:
        dlist = []
        for n in G.nodes():
            if G.out_degree(n) == 0:
                dlist.append(n)
        G.cache["destinations"]["nodes"] = set(dlist)
    return G.cache["destinations"]["nodes"]
# end


def predecessors(G: Graph, n: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Parents of the current node
    """
    assert G.is_directed()
    return set(G.pred[n])
# end
parents = predecessors


def successors(G: Graph, n: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Children of the current node
    """
    assert G.is_directed()
    return set(G.succ[n])
# end
children = successors


def ancestors(G: Graph, v: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Ancestors of the current node
    """
    assert G.is_directed()

    if is_networkx_graph(G):
        return nx.ancestors(G, v)

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
# end


def descendants(G: Graph, u: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Descendants of the current node
    """
    assert G.is_directed()

    if is_networkx_graph(G):
        return nx.descendants(G, u)

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
# end


# ---------------------------------------------------------------------------
# Succesors
# ---------------------------------------------------------------------------

def all_descendants(G: Graph, nodes: NODE_TYPE | Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    """
    Descendants of the list of nodes
    :param G:
    :param nodes: predecessors nodes
    :return: set of successor nodes
    """
    assert G.is_directed()

    if is_instance(nodes, NODE_TYPE):
        nodes = [nodes]

    succ_nodes: set[NODE_TYPE] = set()

    for n in nodes:
        succ_set = successors(G, n)
        succ_nodes.update(succ_set)
    # end

    return succ_nodes
# end


def all_ancestors(G: Graph, nodes: NODE_TYPE | Collection[NODE_TYPE]) -> set[NODE_TYPE]:
    """
    Predecessors of the list of nodes
    :param G:
    :param nodes: successors nodes
    :return: set of predecessor nodes
    """
    assert G.is_directed()

    if is_instance(nodes, NODE_TYPE):
        nodes = [nodes]

    pred_nodes: set[NODE_TYPE] = set()

    for n in nodes:
        prec_set = predecessors(G, n)
        pred_nodes.update(prec_set)
    # end

    return pred_nodes
# end


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
# end


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
