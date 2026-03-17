import networkx as nx
import netx

G = nx.DiGraph()
G.add_edges_from([
    (1, 2), (2, 3),
    (4, 5), (5, 3),
    (1, 4)
])
H = nx.DiGraph()
H.add_edges_from([
    (2, 1), (2, 3),
    (4, 5), (5, 3),
    (1, 4)
])


print(netx.is_markov_equivalent(G, H))
