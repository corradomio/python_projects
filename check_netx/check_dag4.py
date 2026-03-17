from pprint import pprint
import networkx as nx
import netx

G = netx.random_dag(10, 10, connected=True, create_using=nx.DiGraph)
G.add_edge(1,2)
G.add_edge(2,1)

G.add_edge(9,0)
G.add_edge(0,9)

A = netx.adjacency_matrix(G)
# pprint(A)

print(netx.is_partial_adjacency_matrix(A))

netx.draw(A)
netx.show()

# P = netx.power_adjacency_matrix(A)
# pprint(P)
#
# Q = netx.power_adjacency_matrix(A, 16)
# pprint(P)
#
# print((P-Q).sum())
#
# pprint(A * A.T)
# pass

for A in netx.enumerate_directed_graphs(A, True):
    netx.draw(A)
    netx.show()
    # pprint(A)