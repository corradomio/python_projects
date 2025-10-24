import netx as nx


G = nx.DiGraph()
# G.add_edges_from([("X","Y"),("X","B"),("Y","C"), ("A","X"),("A","Y"),("B","Y"),("B","C")])
# G.add_edges_from([("X","Y"), ("Y","Z")])
G.add_edges_from([("X","Y"), ("Z","Y")])
G.add_edges_from([("X","Y"), ("Z","Y"), ("Z","A"), ("B","A")])

print(nx.is_d_separator(G, "X", "Y", set()))
print(nx.is_d_separator(G, "X", "Y", "Z"))
print("---")
print(nx.is_d_separator(G, "X", "Z", set()))
print(nx.is_d_separator(G, "X", "Z", "Y"))
print("---")
print(nx.is_d_separator(G, "X", "B", set()))
print(nx.is_d_separator(G, "X", "B", "Y"))
print(nx.is_d_separator(G, "X", "B", "A"))
print(nx.is_d_separator(G, "X", "B", {"A", "Y"}))
