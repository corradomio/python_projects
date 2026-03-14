import networkx as nx
import iplotx as ipx
import netx

G = nx.DiGraph()
netx.add_edges_from(
    G,[
        # ("A","B","C","D","A"), ("D", "B"),
        # ("A","D","C","B","A"), #("B", "D")
        (0,1,2,3,0), (3, 1),
        (0,3,2,1,0), #("B", "D")
    ]
)

import igraph as ig
import iplotx as ipx

g = ig.Graph.Ring(4)
layout = g.layout("circle").coords
# ipx.network(g, layout)

for H in netx.enumerate_all_directed_graphs(G, True):
    # netx.print_graph_stats(H)
    # netx.draw(H)
    ipx.network(H, layout, vertex_labels=["A","B","C","D"])
    netx.show()
    print(f"... {[e for e in H.edges]}: dag: {nx.is_directed_acyclic_graph(H)}")

