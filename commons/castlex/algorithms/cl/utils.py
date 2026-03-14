import numpy as np
from causallearn.graph.GraphClass import CausalGraph, GeneralGraph


def from_causal_graph(cg: CausalGraph) -> np.ndarray:
    n_nodes = len(cg.G.graph)
    directed = cg.find_fully_directed()
    undirected = cg.find_undirected()
    bidirected = cg.find_bi_directed()

    causal_matrix = np.zeros([n_nodes, n_nodes], dtype=np.int8)
    # for (i, j) in undirected:
    #     causal_matrix[i][j] = 1
    #     causal_matrix[j][i] = 1
    for (i, j) in directed:
        causal_matrix[i][j] = 1
    # for (i, j) in bidirected:
    #     causal_matrix[i][j] = 1
    #     causal_matrix[j][i] = 1
    return causal_matrix


def from_general_graph(gg: GeneralGraph) -> np.ndarray:
    n_nodes = gg.get_num_nodes()
    n_names = gg.get_node_names()
    d_names: dict[str, int] = {
        n_names[i]: i for i in range(n_nodes)
    }

    causal_matrix = np.zeros([n_nodes, n_nodes], dtype=np.int8)

    for e in gg.get_graph_edges():
        n1 = d_names[e.get_node1().get_name()]
        n2 = d_names[e.get_node2().get_name()]
        causal_matrix[n1][n2] = 1

    return causal_matrix
