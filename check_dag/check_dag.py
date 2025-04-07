import networkx as nx
import netx


def main():
    G = netx.random_dag(10, 20, connected=True, create_using=nx.DiGraph)
    D = netx.from_adjacency_matrix(nx.adjacency_matrix(G))

    print(netx.sources(G))

    print("G")
    print(netx.is_netx_graph(G))
    print(netx.is_networkx_graph(G))
    print(netx.is_directed_acyclic_graph(G))

    netx.draw(G)
    netx.show()

    print(nx.ancestors(G, 6))
    print(G.successors(6))
    print(nx.descendants(G, 6))
    print(G.predecessors(6))

    print("D")
    print(netx.is_netx_graph(D))
    print(netx.is_networkx_graph(D))
    print(netx.is_directed_acyclic_graph(D))

    netx.draw(D)
    netx.show()

    print(netx.ancestors(D, 6))
    print(netx.descendants(D, 6))

    pass


if __name__ == "__main__":
    main()
