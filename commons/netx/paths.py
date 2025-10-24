from typing import Iterator, Collection

from .cache import check_cache
from .graph import Graph, NODE_TYPE


# ---------------------------------------------------------------------------
# find_paths
# find_all_paths
# ---------------------------------------------------------------------------

def _find_directed_path(G: Graph, u_path: list[NODE_TYPE], s: NODE_TYPE, v: NODE_TYPE, u_processed: set[NODE_TYPE]) \
        -> Iterator[list[NODE_TYPE]]:

    if s in u_processed:
        return

    if s == v:
        yield u_path + [v]

    for t in G.succ[s]:
        yield from _find_directed_path(G, u_path + [s], t, v, u_processed | {s})
# end


def find_directed_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Find all directed paths between the specified nodes. It returns an iterator

    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: an iterator on uv paths
    """
    assert G.is_directed()

    # if u == v:
    #     return []

    yield from _find_directed_path(G, [], u, v, set())
# end


def find_all_directed_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Find all DIRECTED paths between the specified nodes. It returns the list of paths,
    Note: the implementation is based on 'find_directed_paths(G, u, v)'
    Note: the paths found are cached in 'G.__netx_cache__['directed_paths']

    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: the list of paths
    """

    check_cache(G)

    uv = u, v
    if uv not in G.__netx_cache__["directed_paths"]:
        uv_paths = list(find_directed_paths(G, u, v))
        G.__netx_cache__["directed_paths"][uv] = uv_paths
    return G.__netx_cache__["directed_paths"][uv]
# end


# ---------------------------------------------------------------------------
# find_undirected_paths
# ---------------------------------------------------------------------------

def _find_undirected_path(G: Graph, u_path: Collection[NODE_TYPE], s: NODE_TYPE, v: NODE_TYPE, u_processed: set[NODE_TYPE]) \
        -> Iterator[list[NODE_TYPE]]:
    if s in u_processed:
        return

    if s == v:
        yield u_path + [v]

    for t in G.neighbors(s):
        yield from _find_undirected_path(G, u_path + [s], t, v, u_processed | {s})
# end


def find_undirected_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    yield from _find_undirected_path(G, [], u, v, set())
# end



def find_all_undirected_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Find all UNDIRECTED paths between the specified nodes. It returns the list of paths,
    Note: the implementation is based on 'find_undirected_paths(G, u, v)'
    Note: the paths found are cached in 'G.__netx_cache__['undirected_paths']

    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: the list of paths
    """
    check_cache(G)

    uv = u, v if u < v else v, u
    if uv not in G.__netx_cache__["undirected_paths"]:
        uv_paths = list(find_undirected_paths(G, u, v))
        G.__netx_cache__["undirected_paths"][uv] = uv_paths
    return G.__netx_cache__["undirected_paths"][uv]
# end


# trail: undirectional path
find_all_trails = find_all_undirected_paths

# ---------------------------------------------------------------------------
# all_simple_paths
# ---------------------------------------------------------------------------

def all_simple_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Equivalent of 'nx.all_simple_paths(G, u, v)' based on 'find_all_paths', 'find_all_undirected_paths'
    :param G:
    :param u:
    :param v:
    :return:
    """
    if G.is_directed():
        return find_all_directed_paths(G, u, v)
    else:
        return find_all_undirected_paths(G, u, v)
