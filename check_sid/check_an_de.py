import networkx as nx
import netx


G = netx.add_edges_from(nx.DiGraph(), [
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])

H1 = netx.add_edges_from(nx.DiGraph(), [
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"],
    ["Y1","Y2"]
])

H2 = netx.add_edges_from(nx.DiGraph(), [
    ["X2", "X1"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])


H = netx.add_edges_from(nx.DiGraph(), [
    ["Q","X","B","Y"],
    ["P","X","A"],
    ["P","Y",],
    ["B","F"]
])


for u in H.nodes:
    print(u, ": ANC/_", netx.ancestors(H, u, recursive=False))
    print(u, ": ANC/R", netx.ancestors(H, u, recursive=True))
    print(u, ": DES/_", netx.descendants(H, u, recursive=False))
    print(u, ": DES/R", netx.descendants(H, u, recursive=True))
    print("SRC", netx.sources(H))
    print("DST", netx.destinations(H))