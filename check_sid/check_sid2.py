import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import netx
from gadjid import shd, sid

from itertoolsx import subsets


G=nx.DiGraph()
G.add_edges_from([
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])

H1=nx.DiGraph()
H1.add_edges_from([
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"],
    ["Y1","Y2"]
])

H2=nx.DiGraph()
H2.add_edges_from([
    ["X2", "X1"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])


# netx.draw(G)
# plt.show()
# netx.draw(H1)
# plt.show()
# netx.draw(H2)
# plt.show()

# 2015 - Structural Intervention Distance (SID) for Evaluating Causal Graphs.pdf
def check_2():
    print(nx.structural_intervention_distance(G, G))
    # print(nx.structural_intervention_distance(G, H1))
    # print(nx.structural_intervention_distance(G, H2))


def check_3():

    # for x in G.nodes:
    #     print(f"node[{x}]")
    #     print(f"... PA[{x}]", nx.ancestors(G, x))
    #     print(f"... DE[{x}]", nx.descendants(G, x))

    # print(no_descendats(G, 1, 3, [2]))

    V=set(H2.nodes)

    for x in V:
        for y in V:
            if x == y: continue
            N=V.difference([x,y])
            for Z in subsets(N):
                # print(f"{x}->{y} | {Z}: ", no_descendats(G, x, y, Z))
                print("... ... ...", netx.all_paths_blocked(G, x, y, Z))


def check_4():

    Gm =nx.adjacency_matrix(G).toarray().astype(np.int8)
    H1m=nx.adjacency_matrix(H1).toarray().astype(np.int8)
    H2m=nx.adjacency_matrix(H2).toarray().astype(np.int8)

    # print(shd(Gm, Gm))
    # print(shd(Gm, H1m))
    # print(shd(Gm, H2m))

    print(sid(Gm, Gm,  edge_direction="from row to column"))
    print(sid(Gm, H1m, edge_direction="from row to column"))
    print(sid(Gm, H2m, edge_direction="from row to column"))
    pass



def main():
    # check_2()
    # check_3()
    check_4()
    pass




if __name__ == "__main__":
    main()
