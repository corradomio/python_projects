import networkx as nx
import netx

# G = nx.DiGraph()
# G.add_edges_from([("X", "Y")])
# print(netx.is_d_separated(G, "X", "Y", []))

G = nx.DiGraph()
G.add_edges_from([
    ("X", "A"), ("A", "Y"),
    #("B","Y"),
    ("X", "B"), ("X", "Y"),
    # ("B", "C"),  ("Y", "C")
    ("C", "B"),  ("C", "Y")
])

print(netx.is_directed_acyclic_graph(G))


for P in netx.find_all_undirected_paths(G, "X", "Y"):
    print(P)

print(netx.is_d_separated(G, "X", "Y", ["A"]))