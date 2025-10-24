import matplotlib.pyplot as plt
import numpy as np
import netx
from itertoolsx import subsets
from netx import no_descendats, all_paths_blocked

G = netx.DiGraph().add_edges_from([
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])

H1 = netx.DiGraph().add_edges_from([
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"],
    ["Y1","Y2"]
])

H2 = netx.DiGraph().add_edges_from([
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
    print(netx.structural_intervention_distance(G, G))
    # print(netx.structural_intervention_distance(G, H1))
    # print(netx.structural_intervention_distance(G, H2))


def check_3():

    # for x in G.nodes:
    #     print(f"node[{x}]")
    #     print(f"... PA[{x}]", netx.ancestors(G, x))
    #     print(f"... DE[{x}]", netx.descendants(G, x))

    # print(no_descendats(G, 1, 3, [2]))

    V = set(H2.nodes)

    for x in V:
        for y in V:
            if x == y: continue
            N = V.difference([x,y])
            for Z in subsets(N):
                # print(f"{x}->{y} | {Z}: ", no_descendats(G, x, y, Z))
                print("... ... ...", all_paths_blocked(G, x, y, Z))


def main():
    check_2()
    # check_3()
    pass




if __name__ == "__main__":
    main()
