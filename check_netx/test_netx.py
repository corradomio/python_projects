import netx as nx


def test_graphtype():
    G = nx.Graph()
    assert not G.is_directed()
    assert not G.is_multigraph()
    assert not G.has_loops()

    G = nx.DiGraph()
    assert G.is_directed()
    assert not G.is_multigraph()
    assert not G.has_loops()

    G = nx.MultiGraph()
    assert not G.is_directed()
    assert G.is_multigraph()
    assert not G.has_loops()

    G = nx.MultiDiGraph()
    assert G.is_directed()
    assert G.is_multigraph()
    assert not G.has_loops()

    G = nx.Graph(loops=True)
    assert not G.is_directed()
    assert not G.is_multigraph()
    assert G.has_loops()

    G = nx.DiGraph(loops=True)
    assert G.is_directed()
    assert not G.is_multigraph()
    assert G.has_loops()

    G = nx.DirectAcyclicGraph()
    assert G.is_directed()
    assert not G.is_multigraph()
    assert not G.has_loops()
    assert G.is_acyclic()

    print(G)


def test_graph_properties():
    G = nx.DiGraph()
    assert G.name == "G"

    G = nx.DirectAcyclicGraph(name="DAG", version="1.0")
    assert G.name == "DAG"
    assert G.properties["version"] == "1.0"


def test_graph_compatibility():
    G = nx.DirectAcyclicGraph()

    assert not G.is_multigraph()
    assert G.is_directed()
    assert G.is_acyclic()
    assert not G.has_loops()

    for n in G:
        assert G.has_node(n)
        assert n in G


def test_nodes():
    G = nx.Graph().add_node(1).add_nodes_from([1,2,2,3,3,4,4])

    assert G.order() == 4
    for n in range(1, 5):
        assert n in G

    G.add_edges_from([(1, 2), (1,3),(2,4),(3,4)])
    for n in G:
        assert len(G[n]) == 2

    G = nx.DiGraph().add_edges_from([(1, 2), (1,3),(2,4),(3,4)])

    assert G.order() == 4
    for n in range(1, 5):
        assert n in G

    assert len(G[1]) == 2
    assert len(G[2]) == 1
    assert len(G[3]) == 1
    assert len(G[4]) == 0


def test_add_nodes():
    G = nx.Graph()\
        .add_node(0, name="zero")\
        .add_node(1)\
        .add_node(2, **{"name": "two"})\
        .add_nodes_from([3, (4, {"name": "four"})])

    assert len(G) == 5
    assert G.order() == 5

    for n in range(5):
        assert n in G


def test_degrees_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1, {"info": "01"}), (0, 2), (1, 3, {"info": "13"}), (2, 3)])

    assert G.order() == 4
    assert G.size() == 4

    for n in G.nodes_:
        assert G.degree(n, 2)
        assert G.in_degree(n, 2)
        assert G.out_degree(n, 2)

    assert G.edges[(0, 1)]["info"] == "01"
    assert G.edges[(1, 0)]["info"] == "01"
    assert G.edges[(1, 3)]["info"] == "13"
    assert G.edges[(3, 1)]["info"] == "13"

    assert (0, 1) in G.edges
    assert (1, 0) in G.edges
    assert (1, 3) in G.edges
    assert (3, 1) in G.edges

    assert sorted(G.edges.adj[0]) == [1, 2]
    assert sorted(G.edges.pred[0]) == [1, 2]
    assert sorted(G.edges.succ[0]) == [1, 2]

    assert sorted(G.edges.adj[2]) == [0, 3]
    assert sorted(G.edges.pred[2]) == [0, 3]
    assert sorted(G.edges.succ[2]) == [0, 3]


def test_degrees_digraph():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1, {"info": "01"}), (0, 2), (1, 3, {"info": "13"}), (2, 3)])

    assert G.order() == 4
    assert G.size() == 4

    for n in G.nodes_:
        assert G.degree(n, 2)

    assert G.in_degree(0) == 0
    assert G.out_degree(0) == 2

    assert G.in_degree(3) == 2
    assert G.out_degree(3) == 0

    for n in [1, 2]:
        assert G.in_degree(n) == 1
        assert G.out_degree(n) == 1

    assert G.edges[(0, 1)]["info"] == "01"
    assert G.edges[(1, 3)]["info"] == "13"

    assert (1, 0) not in G.edges
    assert (3, 1) not in G.edges

    assert sorted(G.edges.adj[0]) == [1, 2]
    assert sorted(G.edges.pred[0]) == []
    assert sorted(G.edges.succ[0]) == [1, 2]

    # assert sorted(G.edges.adj[2]) == [0, 3]
    assert sorted(G.edges.pred[2]) == [0]
    assert sorted(G.edges.succ[2]) == [3]


def test_daggen():
    G = nx.random_dag(10, 20)

    for n in range(10):
        assert (n, n) not in G.edges


def test_ancestors_descendants():
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 6)])

    assert nx.sources(G) == {1}
    assert nx.destinations(G) == {4, 5, 6}

    assert nx.ancestors(G, 1) == set()
    assert nx.descendants(G, 1) == {2, 3, 4, 5, 6}

    assert nx.ancestors(G, 2) == {1}
    assert nx.descendants(G, 2) == {4, 5}

    assert nx.ancestors(G, 3) == {1}
    assert nx.descendants(G, 3) == {4, 6}

    assert nx.ancestors(G, 4) == {2, 1, 3}
    assert nx.descendants(G, 4) == set()

    assert nx.ancestors(G, 5) == {2, 1}
    assert nx.descendants(G, 5) == set()

    assert nx.ancestors(G, 6) == {3, 1}
    assert nx.descendants(G, 6) == set()


def test_digraph_add_loop():

    G = nx.DirectAcyclicGraph().add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

    G.add_edge(1, 1)
    assert (1, 1) not in G.edges

    G.add_edge(4, 1)
    assert (4, 1) not in G.edges


def test_all_paths():
    G = nx.DirectAcyclicGraph().add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])
    all_paths = list(nx.find_paths(G, 1, 4))

    assert len(all_paths) == 2
    assert all(map(lambda e: len(e) == 3, all_paths))

    assert all_paths[0] == [1, 2, 4]
    assert all_paths[1] == [1, 3, 4]


def test_neighbors():
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 6)])

    assert G.neighbors(1) == [2, 3]
    assert G.neighbors(3) == [1, 4, 6]

    assert G.neighbors(1, False) == [2, 3]
    assert G.neighbors(3, False) == [1, 4, 6]

    assert G.neighbors(1, True) == [2, 3]
    assert G.neighbors(3, True) == [1, 4, 6]

    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 6)])

    assert G.neighbors(1) == [2, 3]
    assert G.neighbors(3) == [1, 4, 6]

    assert G.neighbors(1, False) == [2, 3]
    assert G.neighbors(3, False) == [4, 6]

    assert G.neighbors(1, True) == []
    assert G.neighbors(3, True) == [1]
