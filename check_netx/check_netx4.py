import networkx as nx
import netx


G = netx.random_dag(10, 10, connected=True, create_using=nx.DiGraph)
netx.draw(G)
netx.show()


print(netx.sources(G))
print(netx.sinks(G))


print("---")
for s in G.nodes:
    print(s, ":", netx.descendants(G, s), ",", netx.ancestors(G, s))
    print(s, ":", nx.descendants(G, s), ",", nx.ancestors(G, s))
    print("---")

