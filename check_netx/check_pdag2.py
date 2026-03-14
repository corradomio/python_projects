import networkx as nx
import netx

G = nx.DiGraph()
G.add_edges_from([
    ("X","A"), ("X","B"),
    ("A","Y"), ("Y","B"),
    ("B","X")
])

print(G.is_directed())
print(netx.is_partial_directed(G))
for H in netx.enumerate_all_directed_graphs(G, dag=False):
    netx.print_graph_stats(H)
    print(f"... edges: {[e for e in G.edges()]}")
    print("... acyclic:", nx.is_directed_acyclic_graph(H))

