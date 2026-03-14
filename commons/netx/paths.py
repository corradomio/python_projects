from typing import Iterator, Collection

from .cache import check_cache
from .graph import Graph, NODE_TYPE


# ---------------------------------------------------------------------------
# is_direct_connected
# is_undirect_connected
# ---------------------------------------------------------------------------

def is_direct_connected(G: Graph, u: NODE_TYPE, v: NODE_TYPE) ->  bool:
    uv_path = next(_find_directed_path(G, [], u, v, set()))
    return uv_path is not None


def is_undirect_connected(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> bool:
    uv_path = next(_find_undirected_path(G, [], u, v, set()))
    return uv_path is not None


# ---------------------------------------------------------------------------
# find_directed_paths
# find_all_directed_paths (compatibility)
# ---------------------------------------------------------------------------

def _find_directed_path(G: Graph, u_path: list[NODE_TYPE], s: NODE_TYPE, v: NODE_TYPE, u_processed: set[NODE_TYPE]) \
        -> Iterator[list[NODE_TYPE]]:
    assert s not in u_processed

    if s == v:
        yield u_path + [v]

    for t in G.succ[s]:
        if t in u_processed: continue
        yield from _find_directed_path(G, u_path + [s], t, v, u_processed | {s})
# end


def find_directed_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Find all DIRECTED paths between the specified nodes. It returns the list of paths,
    Note: the implementation is based on 'find_directed_paths(G, u, v)'
    Note: the paths found are cached in 'G.__netx_cache__['directed_paths']

    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: the list of paths
    """
    assert G.is_directed()

    # support for networkx
    check_cache(G)

    uv = u, v
    if uv not in G.__netx_cache__["directed_paths"]:
        uv_paths = list(_find_directed_path(G, [], u, v, set()))
        G.__netx_cache__["directed_paths"][uv] = uv_paths
    return G.__netx_cache__["directed_paths"][uv]
# end


# compatibility
find_all_directed_paths = find_directed_paths


# ---------------------------------------------------------------------------
# find_undirected_paths
# find_all_undirected_paths (compatibility)
# find_all_trails (compatibility)
# ---------------------------------------------------------------------------

def all_neighbors(G: Graph, u: NODE_TYPE) -> set[NODE_TYPE]:
    """
    Equivalent to 'nx.all_neighbors(G, u, v)' but caches the results
    :param G: graph node
    :param u: set of all neighbours
    :return:
    """
    if u not in G.__netx_cache__["undirected_neighbors"]:
        if G.is_directed():
            undirected_neighbors = set()
            undirected_neighbors.update(v for v in G.pred[u])
            undirected_neighbors.update(v for v in G.succ[u])
        else:
            undirected_neighbors = set()
            undirected_neighbors.update(v for v in G.adj[u])
        G.__netx_cache__["undirected_neighbors"][u] = undirected_neighbors
    return G.__netx_cache__["undirected_neighbors"][u]
# end


def _find_undirected_path(G: Graph, u_path: Collection[NODE_TYPE], s: NODE_TYPE, v: NODE_TYPE, u_processed: set[NODE_TYPE]) \
        -> Iterator[list[NODE_TYPE]]:
    assert s not in u_processed

    if s == v:
        yield u_path + [v]

    for t in all_neighbors(G, s):
        if t in u_processed: continue
        yield from _find_undirected_path(G, u_path + [s], t, v, u_processed | {s})
# end


def find_undirected_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Find all UNDIRECTED paths between the specified nodes. It returns the list of paths,
    Note: the implementation is based on 'find_undirected_paths(G, u, v)'
    Note: the paths found are cached in 'G.__netx_cache__['undirected_paths']

    :param G: graph to analyze
    :param u: start node
    :param v: destination node
    :return: the list of paths
    """
    # support for networkx
    check_cache(G)

    uv = u, v if u < v else v, u
    if uv not in G.__netx_cache__["undirected_paths"]:
        uv_paths = list(_find_undirected_path(G, [], u, v, set()))
        uv_paths = sorted(uv_paths, key=lambda l:len(l))
        G.__netx_cache__["undirected_paths"][uv] = uv_paths
    return G.__netx_cache__["undirected_paths"][uv]
# end


# compatibility
find_all_undirected_paths = find_undirected_paths

# trail: undirectional path
find_all_trails = find_all_undirected_paths


# ---------------------------------------------------------------------------
# all_simple_paths
# ---------------------------------------------------------------------------

def all_simple_paths(G: Graph, u: NODE_TYPE, v: NODE_TYPE) -> Iterator[list[NODE_TYPE]]:
    """
    Equivalent of 'nx.all_simple_paths(G, u, v)' based on 'find_directed_paths', 'find_all_undirected_paths'
    :param G: graph
    :param u: source node
    :param v: destination node
    :return: list of paths
    """
    if G.is_directed():
        return find_directed_paths(G, u, v)
    else:
        return find_all_undirected_paths(G, u, v)


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
