import netx
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([
    ("X","A"), ("X","B"),
    ("A","Y"), ("Y","B")
])

print("paths")
for P in netx.find_all_undirected_paths(G, "X", "Y"):
    print(P)

for P in netx.find_all_directed_paths(G, "X", "Y"):
    print(P)

