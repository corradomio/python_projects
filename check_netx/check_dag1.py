import networkx as nx
import netx


for G in netx.enumerate_all_directed_acyclic_graphs(3, create_using=nx.DiGraph):
    netx.print_graph_stats(G)
    print(f"... edges: {[e for e in G.edges()]}")
    print("... acyclic:", nx.is_directed_acyclic_graph(G))