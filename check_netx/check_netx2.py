import networkx as nx
import netx

def _create_graphs():
    G = nx.DiGraph()
    G.add_edges_from([
        (1, 2), (2, 3),
        (4, 5), (5, 3),
        (1, 4)
    ])
    return G


def test_path_chains():
    G = _create_graphs()
    # print(netx.path_chains(G, set(G.nodes())))
    assert netx.path_chains(G, [1,2,3]) == {2}


def test_path_forks():
    G = _create_graphs()
    # print(netx.path_forks(G, set(G.nodes())))
    assert netx.path_forks(G, set(G.nodes())) == {1}


def test_path_colliders():
    G = _create_graphs()
    # print(netx.path_colliders(G, set(G.nodes())))
    assert netx.path_colliders(G, set(G.nodes())) == {3}


def main():
    test_path_chains()
    test_path_forks()
    test_path_colliders()


if __name__ == '__main__':
    main()
