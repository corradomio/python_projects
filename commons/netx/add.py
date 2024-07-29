from typing import Union

import networkx as nx


def add_nodes(G: Union[nx.Graph, nx.DiGraph], nodes: list, **kwargs) -> Union[nx.Graph, nx.DiGraph]:
    G.add_nodes_from(nodes, **kwargs)
    return G


def add_edges(G: Union[nx.Graph, nx.DiGraph], edges: list, **kwargs) -> Union[nx.Graph, nx.DiGraph]:
    G.add_edges_from(edges, **kwargs)
    return G