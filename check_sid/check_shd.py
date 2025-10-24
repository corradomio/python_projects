import matplotlib.pyplot as plt
import numpy as np
import netx


G = netx.DiGraph(name="G").add_edges_from([
        ["X1", "X2"],
        ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
        ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
    ])

H1 = netx.DiGraph(name="H1").add_edges_from([
        ["X1", "X2"],
        ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
        ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"],
        ["Y1","Y2"]
    ])

H2 = netx.DiGraph(name="H2").add_edges_from([
        ["X2", "X1"],
        ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
        ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
    ])


# def check_1():
#     G = netx.random_dag(10, 10, create_using=netx.DiGraph)
#     H = netx.random_dag(10, 10, create_using=netx.DiGraph)
#     netx.draw(G)
#     plt.show()
#     netx.draw(H)
#     plt.show()
#     Mg = netx.adjacency_matrix(G)
#     Mh = netx.adjacency_matrix(H)
#     print(netx.structural_hamming_distance(G, H))
#     print(netx.structural_hamming_distance(Mg, Mh))
#     print(netx.structural_hamming_distance(G, G))
#
#     Mr = np.transpose(Mg)
#     R = netx.from_numpy_matrix(Mr)
#     netx.draw(R)
#     plt.show()
#     print(netx.structural_hamming_distance(Mg, Mr))

# 2015 - Structural Intervention Distance (SID) for Evaluating Causal Graphs.pdf
def check_2():
    # print(G.order(), list(G.nodes))
    # print(H1.order(), list(H1.nodes))
    # print(H2.order(), list(H2.nodes))

    # netx.draw(G)
    # plt.show()
    # netx.draw(H1)
    # plt.show()
    # netx.draw(H2)
    # plt.show()

    # print(netx.structural_hamming_distance(G, H1))
    # print(netx.structural_hamming_distance(G, H2))

    # print("G,G ", netx.structural_intervention_distance(G, G))
    # print("G,H1", netx.structural_intervention_distance(G, H1))
    print("G,H2", netx.structural_intervention_distance(G, H2))

def main():
    # check_1()
    check_2()
    pass




if __name__ == "__main__":
    main()
