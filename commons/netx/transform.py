from typing import List, Dict, Set

import networkx as nx
from networkx import Graph


def coarsening_graph(g: Graph, partitions: List[List[int]], create_using=None, direct=False, **kwargs) -> Graph:
    """
    Create a coarse graph using the partitions as 'super' nodes and edges between
    partitions i and j if there exist a node in partition i connected to a node in partition j

    :param g: original graph
    :param partitions: partitions
    :param create_using: which graph to populate, or None
    :param direct: if to create a direct graph
    :param kwargs: extra parameters passed to the created graph
    :return: a coarsed graph
    """
    assert isinstance(partitions, list)

    # in_partition[u] -> partition of u
    def in_partition(nv):
        in_part = [0]*nv
        c = 0
        for partition in partitions:
            for u in partition:
                in_part[u] = c
            c += 1
        return in_part

    # create an empty graph
    if create_using is not None:
        coarsed = create_using
    elif direct:
        coarsed = nx.DiGraph(**kwargs)
    else:
        coarsed = nx.Graph(**kwargs)
    name = g.graph["name"] if "name" in g.graph else "G"
    # if "name" not in coarsed.graph:
    coarsed.graph["name"] = f"coarsed-{name}"

    # force partitions to be a list
    partitions = list(partitions)
    in_part = in_partition(g.order())
    for u, v in list(g.edges):
        cu = in_part[u]
        cv = in_part[v]
        if cu != cv:
            coarsed.add_edge(cu, cv)
    # end
    return coarsed
# end


def closure_coarsening_graph(g: nx.DiGraph, create_using=None, direct=True, **kwargs) -> nx.Graph:
    """
    Create a coarse graph using the following protocol

        1) for each node creates the transitive closure
        2) create an edge from closure i and closure j if closure i is a proper subset of closure j
        3) apply a transitive reduction

    :param g: original graph
    :param create_using: which graph to populate, or None
    :param direct: if to create a direct graph
    :param kwargs: extra parameters passed to the created graph
    :return: a coarsed graph
    """
    assert g.is_directed()

    def closure_of(u: int, closures: Dict[int, Set[int]]) -> Set[int]:
        visited: Set[int] = set()
        tovisit: List[int] = [u]
        while len(tovisit) > 0:
            u: int = tovisit.pop()
            if u in visited:
                continue

            if u in closures:
                visited.update(closures[u])
            else:
                visited.add(u)
                tovisit += list(g.succ[u])
        # end
        return visited
    # end

    def is_subset_of(s1: Set[int], s2: Set[int]) -> bool:
        if len(s1) >= len(s2):
            return False
        for u in s1:
            if u not in s2:
                return False
        return True
    # end

    # create an empty graph
    if create_using is not None:
        coarsed = create_using
    elif direct:
        coarsed = nx.DiGraph(**kwargs)
    else:
        coarsed = nx.Graph(**kwargs)
    name = g.graph["name"] if "name" in g.graph else "G"
    coarsed.graph["name"] = f"closure-coarsed-{name}"

    # compute the closures
    closures: Dict[int, Set[int]] = dict()
    for u in g:
        closures[u] = closure_of(u, closures)
    # end

    # compute the simplified graph
    for u in closures:
        cu = closures[u]
        for v in closures:
            cv = closures[v]

            if len(cu) == len(cv):
                continue
            if is_subset_of(cv, cu):
                coarsed.add_edge(u, v)
        # end
    # end

    # apply the transitive reduction
    reducted = nx.transitive_reduction(coarsed)


    reducted.graph.update(coarsed.graph)

    return coarsed
# end


def print_graph_stats(g:nx.Graph):
    n = len(g.order())
    m = len(g.size())
    if g.is_directed():
        print(f"G={{V: {n}, E: {m}}}")
    else:
        print(f"G={{V: {n}, E: {m}}}, directed")
# end
