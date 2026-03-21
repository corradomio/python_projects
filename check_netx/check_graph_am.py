import networkx as nx
import netx


def main():
    A = netx.random_adjacency_matrix(10, 10, directed=True)
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    netx.draw(G)
    netx.show()

    print(list(netx.connected_components_adjacency_matrix(A)))

    for i in range(10):
        print(f"node {i}")
        print(f"... pa", netx.ancestors_adjacency_matrix(A, i, False))
        print(f"... ch", netx.descendants_adjacency_matrix(A, i, False))
        print(f"... an", netx.ancestors_adjacency_matrix(A, i))
        print(f"... de", netx.descendants_adjacency_matrix(A, i))
        pass

    print(" --> ")
    for i in range(10):
        for j in range(10):
            if i==j: continue
            for p in netx.paths_adjacency_matrix(A, i, j, undirected=False):
                print(f"{i}->{j}: {p}")
                pass
    print(" --- ")
    for i in range(10):
        for j in range(10):
            if i==j: continue
            for p in netx.paths_adjacency_matrix(A, i, j, undirected=True):
                print(f"{i}--{j}: {p}")
                pass
    print(" ::: ")
    pass


if __name__ == "__main__":
    main()

