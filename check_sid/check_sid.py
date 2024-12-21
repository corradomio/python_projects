from typing import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import netx



def main():

    G = netx.random_dag(10, 10, create_using=netx.DiGraph())
    print(G.nodes)
    print(G.nodes())
    print(G.edges)
    # print(G.edges())

    u,v = list(G.edges)[0]
    print(f"add edge {(v,u)}")
    G.add_edge(v, u)

    print(nx.is_directed(G))
    print(netx.is_dag(G))
    print(netx.is_partial_directed(G))
    print(netx.undirect_edges(G))

    for D in netx.pdag_enum(G):
        netx.util.draw(D)
        plt.show()
    pass


if __name__ == "__main__":
    main()
