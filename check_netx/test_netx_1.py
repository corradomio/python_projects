import networkx as nx
import netx
import numpy as np


def _create_graphs():
    G1 = nx.DiGraph()
    G1.add_nodes_from([0,1,2])
    G1.add_edges_from([(0, 1),(0,2),(1,2)])

    G2 = netx.DiGraph()
    G2.add_nodes_from([0,1,2])
    G2.add_edges_from([(0, 1),(0,2),(1,2)])

    return G1, G2

def test_graph_type():
    # Graph, DiGraph, MultiGraph, MultiDiGraph
    G1, G2 = _create_graphs()

    am1: np.ndarray = nx.adjacency_matrix(G1).toarray()
    am2: np.ndarray = netx.adjacency_matrix(G2)

    assert np.abs(am1-am2).sum() == 0

def test_graph_node():
    G1, G2 = _create_graphs()

    n1 = {n for n in G1}
    n2 = {n for n in G2}

    assert n1 == n2

    for n in range(3):
        assert n in G1
        assert n in G2

    assert len(G1) == len(G2)

    n1 = {n for n in G1.nodes}
    n2 = {n for n in G2.nodes}

    assert n1 == n2

    n1 = {n for n in G1.nodes()}
    n2 = {n for n in G2.nodes()}

    assert n1 == n2

def test_graph_degree():
    G1, G2 = _create_graphs()

    for n in range(3):
        assert G1.degree[n] == G2.degree[n]

    for n in range(3):
        assert G1.in_degree[n] == G2.in_degree[n]

    for n in range(3):
        assert G1.out_degree[n] == G2.out_degree[n]

    assert len(G1) == len(G2)
