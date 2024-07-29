import networkx as nx


def print_graph_stats(g: nx.Graph):
    n = g.order()
    m = g.size()
    if g.is_directed():
        print(f"G={{|V|={n}, |V|={m}, direct}}")
    else:
        print(f"G={{|V|={n}, |V|={m}}}")
# end
