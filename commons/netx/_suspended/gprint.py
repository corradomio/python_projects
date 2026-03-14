import networkx as nx


def print_graph_stats(G: nx.Graph):
    n = G.order()
    m = G.size()
    if G.is_directed():
        print(f"G={{|V|={n}, |V|={m}, direct}}")
    else:
        print(f"G={{|V|={n}, |V|={m}}}")
# end
