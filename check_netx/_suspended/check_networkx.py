import networkx as nx
import netx as nxx

g = nx.Graph()

g = nx.DiGraph()
g.add_edge(1, 2)

print(nx.is_directed_acyclic_graph(g))
