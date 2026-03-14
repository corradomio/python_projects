import networkx as nx
import netx

G = netx.random_dag(10, 10, create_using=nx.DiGraph)
netx.draw(G)
netx.show()

nodes = list(G.nodes())
n_nodes = len(nodes)

# for i in range(n_nodes):
#     for j in range(i+1, n_nodes):
#         u = nodes[i]
#         v = nodes[j]
#         print(f"{u}---{v}:")
#         for P in netx.find_all_undirected_paths(G, u, v):
#             print("...", P)


for i in range(n_nodes):
    for j in range(i+1, n_nodes):
        u = nodes[i]
        v = nodes[j]
        print(f"d-separated({u}, {v}): {netx.is_d_separated(G, u, v, set())}")

