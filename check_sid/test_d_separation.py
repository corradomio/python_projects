import netx

def test_chain1():
    # 1->2->3
    G = netx.DiGraph()
    G.add_edges_from([(1,2,3)])

    assert netx.all_paths_blocked(G, 1,3,[2])

def test_chain2():
    # 1->2->3->4
    G = netx.DiGraph()
    G.add_edges_from([(1,2,3, 4)])

    assert netx.all_paths_blocked(G, 1,4,[2])
    assert netx.all_paths_blocked(G, 1,4,[3])


def test_fork1():
    # 1<-2->3
    G = netx.DiGraph()
    G.add_edges_from([(2,1),(2,3)])

    assert netx.all_paths_blocked(G, 1,3,[2])


def test_fork2():
    # 1<-2->3<-4->5
    G = netx.DiGraph()
    G.add_edges_from([(2,1),(2,3),(4,3),(4,5)])

    assert netx.all_paths_blocked(G, 1,5,[2])
    assert netx.all_paths_blocked(G, 1,5,[4])
    assert netx.all_paths_blocked(G, 1,5,[2,4])

def test_collider_1():
    # 1->2<-3
    G = netx.DiGraph()
    G.add_edges_from([(1,2), (3,2)])

    assert netx.all_paths_blocked(G, 1, 3, [])

def test_collider_2():
    # 1->2<-3->4<-5
    G = netx.DiGraph()
    G.add_edges_from([(1, 2), (3, 2),(3,4),(5,4)])

    assert netx.all_paths_blocked(G, 1, 5, [3])


# Causal Inference and Discovery
def test_graph_cid_62():
    G = netx.DiGraph()
    G.add_edges_from([("X", "A"), ("X", "B"), ("A","Y"), ("Y","B")])

    assert netx.all_paths_blocked(G, "X", "Y", ["A"])


# Causal Inference and Discovery
# X->A<-B->Y        In the book the image IS WRONG
def test_graph_cid_63():
    G = netx.DiGraph()
    G.add_edges_from([("A","A"),("B","A"),("B", "Y")])

    assert netx.all_paths_blocked(G, "X", "Y", [])
    assert netx.all_paths_blocked(G, "X", "Y", ["B"])


# Causal Inference and Discovery
def test_graph_cid_64():
    G = netx.DiGraph()
    G.add_edges_from([("X","Y"),("X","B"),("Y","C"), ("A","X"),("A","Y"),("B","Y"),("B","C")])

    assert netx.all_paths_blocked(G, "X", "Y", ["A", "B", "C"])
