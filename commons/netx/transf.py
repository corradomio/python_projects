#
# It used .graph.Graph, a better(?) graph representation
#
from .graph import Graph, NODE_TYPE


def replace_nodes(G: Graph, n: NODE_TYPE, neighbors: list[NODE_TYPE], r: NODE_TYPE) -> Graph:
    """
    Replace 'nodes' with 'r'
    :param G: graph to modify
    :param n: previous node
    :param neighbors: nodes to replace
    :param r: node used as replacement
    :return: updated graph
    """
    assert G.is_directed(), "Only for directed graphs"

    G.add_node(r)
    G.add_edge(n, r)

    for u in neighbors:
        ulist = G.neighbors(u, inbound=False)
        for v in ulist:
            eprops: dict = G._edges[(u, v)]
            G.add_edge(r, v, **eprops)

    G.remove_edges_from([(n, v) for v in neighbors])
    G.remove_singletons()
    return G
# end
