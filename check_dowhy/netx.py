import networkx as nx

def to_gml(G, stringizer):
    return "\n".join(list(nx.generate_gml(G, stringizer)))