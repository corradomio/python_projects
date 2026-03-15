import networkx as nx
import netx

def assert_dict_eq(d1: dict, d2: dict):
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert len(d1) == len(d2)
    for k in d1:
        assert k in d2
        assert d1[k] == d2[k]


def _create_graph():
    G = nx.gnm_random_graph(10, 20)
    H = netx.NetxGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    return G, H

def _create_empty_graph():
    G = nx.Graph()
    H = netx.NetxGraph()
    return G, H

def _create_empty_digraph():
    G = nx.DiGraph()
    H = netx.NetxDiGraph()
    return G, H

def _create_digraph():
    G = nx.gnr_graph(10, 0.2, create_using=nx.DiGraph)
    H = netx.NetxDiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    return G, H


def test_order_size():
    G, H = _create_graph()
    assert G.number_of_nodes() == H.number_of_nodes()
    assert G.order() == H.order()
    assert G.number_of_edges() == H.number_of_edges()
    assert G.size() == H.size()
    assert len(G) == len(H)


def test_nodes():
    G, H = _create_graph()

    for n in G:
        assert n in H
        assert H.has_node(n)

    for n in G.nodes:
        assert H.has_node(n)

    for n in G.nodes():
        assert H.has_node(n)


def test_edges():
    G, H = _create_graph()

    for u, v in G.edges():
        assert u in H
        assert v in H
        assert H.has_edge(u, v)


def test_add_node():
    G, H = _create_empty_graph()
    G.add_node(1)
    H.add_node(1)

    assert 1 in G and 1 in H

    G.add_nodes_from([2,3])
    H.add_nodes_from([2,3])

    for n in G.nodes:
        assert n in H


def test_add_node_props():
    G, H = _create_empty_graph()

    G.add_node(1, color="yellow")
    G.add_nodes_from([(2, {"color":"blu"}),3])
    G.add_nodes_from((4, 5), **{"color": "green"})
    G.add_nodes_from([(6, {"weight":0}),3], **{"color": "red"})

    H.add_node(1, color="yellow")
    H.add_nodes_from([(2, {"color":"blu"}),3])
    H.add_nodes_from((4, 5), **{"color": "green"})
    H.add_nodes_from([(6, {"weight":0}),3], **{"color": "red"})

    for n in G.nodes:
        gnp = G.nodes[n]
        hnp = H.nodes[n]
        assert_dict_eq(gnp, hnp)


def test_add_edge_props():
    G, H = _create_empty_graph()

    G.add_edge(3, 4, color="yellow")
    H.add_edge(3, 4, color="yellow")

    G.add_edges_from([(1,2), (2,3)], color="red")
    H.add_edges_from([(1,2), (2,3)], color="red")

    # NOT supported in networkx AND netx
    # G.add_edges_from([((3,1), {"weight": 0})], color="blue")
    # H.add_edges_from([((3,1), {"weight": 0})], color="blue")

    for u, v in G.edges:
        gep = G.edges[u, v]
        hep = H.edges[u, v]
        assert_dict_eq(gep, hep)


def test_neighbors_graph():
    G, H = _create_graph()

    for n in G.nodes:
        gnn = sorted(G.neighbors(n))
        hnn = sorted(H.neighbors(n))
        assert gnn == hnn
        pass
    pass


def test_neighbors_digraph():
    G, H = _create_digraph()

    for n in G.nodes:
        gnn = sorted(G.neighbors(n))
        # in nx 'neighbors' is equivalent to 'successors'
        # in netx it is possible to specify the direction:
        #   inbound = None, False, True
        hnn = sorted(H.neighbors(n))
        assert gnn == hnn

        gnn = sorted(G.successors(n))
        hnn = sorted(H.successors(n))
        assert gnn == hnn

        gnn = sorted(G.predecessors(n))
        hnn = sorted(H.predecessors(n))
        assert gnn == hnn
        pass
    pass


def test_degree_graph():
    G, H = _create_graph()

    for n in G.nodes:
        assert G.degree(n) == H.degree(n)
        assert G.degree[n] == H.degree[n]
    pass


def test_degree_digraph():
    G, H = _create_digraph()

    for n in G.nodes:
        assert G.degree(n) == H.degree(n)
        assert G.degree[n] == H.degree[n]

        assert G.in_degree(n) == H.in_degree(n)
        assert G.in_degree[n] == H.in_degree[n]

        assert G.out_degree(n) == H.out_degree(n)
        assert G.out_degree[n] == H.out_degree[n]
    pass


def test_edges_graph():
    G, H = _create_graph()

    for e in G.edges:
        assert e in H.edges
        assert e[::-1] in H.edges


def test_edges_digraph():
    G, H = _create_digraph()

    for e in G.edges:
        assert e in H.edges


def test_adj_props_graph():
    G, H = _create_empty_graph()

    G.add_edge(3, 4, color="yellow")
    H.add_edge(3, 4, color="yellow")

    G.add_edges_from([(1, 2), (2, 3)], color="red")
    H.add_edges_from([(1, 2), (2, 3)], color="red")

    nodes = list(G.nodes)

    for u in nodes:
        for v in nodes:
            assert G.has_edge(u, v) == H.has_edge(u, v)
            if G.has_edge(u, v):
                guv = G.adj[u][v]
                huv = H.adj[u][v]
                assert_dict_eq(guv, huv)

    pass


def test_adj_props_digraph():
    G, H = _create_empty_digraph()

    G.add_edge(3, 4, color="yellow")
    H.add_edge(3, 4, color="yellow")

    G.add_edges_from([(1, 2), (2, 3)], color="red")
    H.add_edges_from([(1, 2), (2, 3)], color="red")

    nodes = list(G.nodes)

    for u in nodes:
        for v in nodes:
            assert G.has_edge(u, v) == H.has_edge(u, v)
            if G.has_edge(u, v):
                guv = G.adj[u][v]
                huv = H.adj[u][v]
                assert_dict_eq(guv, huv)

    for u in nodes:
        for v in nodes:
            assert G.has_edge(u, v) == H.has_edge(u, v)
            if G.has_edge(u, v):
                guv = G.succ[u][v]
                huv = H.succ[u][v]
                assert_dict_eq(guv, huv)

    for u in nodes:
        for v in nodes:
            assert G.has_edge(u, v) == H.has_edge(u, v)
            if G.has_edge(u, v):
                guv = G.pred[v][u]
                huv = H.pred[v][u]
                assert_dict_eq(guv, huv)

    pass


def test_graph_views():
    G, H = _create_graph()

    gnv = G.nodes
    gadjv = G.adj
    gev = G.edges()
    gdv = G.degree()

    # assert gnv is G.nodes()
    # assert gev is G.edges()
    # assert gdv is G.degree()

    hnv = H.nodes
    hadjv = H.adj
    hev = H.edges
    hdv = H.degree

    # assert hnv is H.nodes()
    # assert hadjv is H.adj()
    # assert hev is H.edges()
    # assert gdv is H.degree()

    pass

def test_digraph_views():
    G, H = _create_digraph()

    gnv = G.nodes
    gadjv = G.adj
    gpredv = G.pred
    gsuccv = G.succ
    gev = G.edges
    giev = G.in_edges
    goev = G.out_edges
    gdv = G.degree
    gidv = G.in_degree
    godv = G.out_degree

    # assert gnv is G.nodes()
    # assert gev is G.edges()
    # assert giev is G.in_edges()
    # assert goev is G.out_edges()
    # assert gdv is G.degree()
    # assert gidv is G.in_degree()
    # assert godv is G.out_degree()

    hnv = H.nodes
    hadjv = H.adj
    hpredv = H.pred
    hsuccv = H.succ
    hev = H.edges
    hiev = H.in_edges
    hoev = H.out_edges
    hdv = H.degree
    hidv = H.in_degree
    hodv = H.out_degree

    # assert hnv is H.nodes()
    # assert hadjv is H.adj(1)
    # assert hpredv is H.pred(1)
    # assert hsuccv is H.succ(1)
    # assert hev is H.edges()
    # assert hiev is H.in_edges()
    # assert hoev is H.out_edges()
    # assert hdv is H.degree()
    # assert hidv is H.in_degree()
    # assert hodv is H.out_degree()

    r = H.nodes()
    r = H.adj()
    r = H.pred()
    r = H.succ()
    r = H.edges()
    r = H.in_edges()
    r = H.out_edges()
    r = H.degree()
    r = H.in_degree()
    r = H.out_degree()

    # -- nodes
    g = G.nodes[1]
    h = H.nodes[1]

    # -- adj
    g = G.adj[1]
    h = H.adj[1]

    g = G.adj[1][0]
    h = H.adj[1][0]

    # 1->0

    # -- succ
    g = G.succ[1]
    h = H.succ[1]

    g = G.succ[1][0]
    h = H.succ[1][0]

    # -- pred
    g = G.pred[0]
    h = H.pred[0]

    g = G.pred[0][1]
    h = H.pred[0][1]

    pass


if __name__ == "__main__":
    test_order_size()
    test_nodes()
    test_edges()
    test_add_node()
    test_add_node_props()
    test_add_edge_props()
    test_neighbors_graph()
    test_neighbors_digraph()
    test_degree_graph()
    test_degree_digraph()
    test_edges_graph()
    test_edges_digraph()

    test_adj_props_graph()
    test_adj_props_digraph()
    test_graph_views()
    test_digraph_views()
    pass

