import netx as nx


def main():
    # G: nx.Graph = nx.random_dag(20, 100)
    # print(nx.ancestors(G, 6))
    # print(nx.descendants(G, 2))
    # print(nx.sources(G))
    # print(nx.destinations(G))

    G = nx.DiGraph().add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 6), (5, 7), (4, 7),
                                     (3, 1)])

    for path in nx.find_paths(G, 1, 7):
        print(path)

    print(G[1])
    print(G[7])

    G = nx.Graph().add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (2, 5), (3, 6), (5, 7), (4, 7),
                                   (3, 1)])

    print(G[1])
    print(G[7])

    print(G[(1,2)])



if __name__ == '__main__':
    main()
