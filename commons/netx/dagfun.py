from collections import deque
from typing import Collection, Union, Iterator
import networkx as nx
from stdlib import is_instance
from .cache import check_cache
from .gclass import is_networkx_graph
from .graph import Graph, NODE_TYPE, EDGE_TYPE


# ---------------------------------------------------------------------------
# add_edges_from
# ---------------------------------------------------------------------------

def add_edges_from(G, elist: list[EDGE_TYPE] | Iterator[EDGE_TYPE], **eprops) -> Graph:
    """
    Add support for edges (u1,u2) od (u,v,uvprops)  and paths (u1,...,un) or (u1,...,un,uprops)
    :param G: graph
    :param elist: list of edges/paths
    :param eprops: common properties
    :return: graph G updated
    """
    for e in elist:
        if len(e) == 2:
            u, v = e
            G.add_edge(u, v, **eprops)
            continue

        if isinstance(e[-1], dict):
            uvprops = e[-1]
            e = e[:-1]
        else:
            uvprops = {}

        n = len(e)
        for i in range(n - 1):
            u = e[i]
            v = e[i + 1]
            G.add_edge(u, v, **(eprops | uvprops))
    return G
# end


#
#   is_directed_acyclic_graph
#   sources
#   destinations
#
#   predecessors == parents   (single step)
#   succesors    == children     (single step)
#
#   ancestors   (recursive)
#   descendants (recursive)
#
#   find_paths
#   find_all_directed_paths
#   find_all_undirected_paths
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
is_dag = is_directed_acyclic_graph


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
#   sources
#   destinations

def sources(G: Graph) -> set[NODE_TYPE]:
    """
    Direct Graph sources: nodes with 0 in-degree
    """
    assert G.is_directed()

    # support for networkx
    check_cache(G)

    if "nodes" not in G.__netx_cache__["sources"]:
        sset = set()
        for n in G.nodes():
            if G.in_degree(n) == 0:
                sset.add(n)

        G.__netx_cache__["sources"]["nodes"] = sset
    return G.__netx_cache__["sources"]["nodes"]
# end


def destinations(G: Graph) -> set[NODE_TYPE]:
    """
    Direct Graph destinations: nodes with 0 out-degree
    """
    assert G.is_directed()

    # support for networkx
    check_cache(G)

    if "nodes" not in G.__netx_cache__["destinations"]:
        dset = set()
        for n in G.nodes():
            if G.out_degree(n) == 0:
                dset.add(n)
        G.__netx_cache__["destinations"]["nodes"] = dset
    return G.__netx_cache__["destinations"]["nodes"]
# end


# ---------------------------------------------------------------------------
# descendants
# ancestors
# ---------------------------------------------------------------------------
# multimethod unable to support the signatures:
#
#   def descendants(G: Graph, nodes:Collection[NODE_TYPE], recursive: bool = False) -> set[NODE_TYPE]:
#   def descendants(G: Graph, n: NODE_TYPE, recursive: bool = False) -> set[NODE_TYPE]:

def descendants(G: Graph, nodes: Union[NODE_TYPE, Collection[NODE_TYPE]], recursive=False) -> set[NODE_TYPE]:
    assert G.is_directed()

    if is_instance(nodes, NODE_TYPE):
        nodes = [nodes]

    D = set()
    if recursive:
        for u in nodes:
            D = D.union(_descendants(G, u))
    else:
        for u in nodes:
            D = D.union(G.succ[u])
    return D
# end


def ancestors(G: Graph, nodes: Union[NODE_TYPE, Collection[NODE_TYPE]], recursive=False) -> set[NODE_TYPE]:
    assert G.is_directed()

    if is_instance(nodes,NODE_TYPE):
        nodes = [nodes]

    A = set()
    if recursive:
        for v in nodes:
            A = A.union(_ancestors(G, v))
    else:
        for v in nodes:
            A = A.union(G.pred[v])
    return A
# end

# ---------------------------------------------------------------------------
# _descendants
# _ancestors
# ---------------------------------------------------------------------------

def _descendants(G: Graph, u: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Direct Graph: descendants of the current node (recursive successors)
    """
    assert G.is_directed()

    # support for networkx
    check_cache(G)

    if is_networkx_graph(G):
        return nx.descendants(G, u)

    if u not in G.__netx_cache__["descendants"]:
        waiting = deque()
        descendants: set[NODE_TYPE] = set()
        waiting.extend(G.succ[u])
        while waiting:
            v = waiting.popleft()
            if v not in descendants:
                descendants.add(v)
                waiting.extend(G.succ[v])
        G.__netx_cache__["descendants"][u] = descendants
    # end
    return G.__netx_cache__["descendants"][u]
# end


def _ancestors(G: Graph, v: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Direct Graph: ancestors of the current node (recursive predecessors)
    """
    assert G.is_directed()

    # support for networkx
    check_cache(G)

    if is_networkx_graph(G):
        return nx.ancestors(G, v)

    if v not in G.__netx_cache__["ancestors"]:
        waiting = deque()
        ancestors: set[NODE_TYPE] = set()
        waiting.extend(G.pred[v])
        while waiting:
            u = waiting.popleft()
            if u not in ancestors:
                ancestors.add(u)
                waiting.extend(G.pred[u])
        G.__netx_cache__["ancestors"][v] = ancestors
    # end
    return G.__netx_cache__["ancestors"][v]
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
