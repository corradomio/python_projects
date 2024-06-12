from netx import Graph


def main():
    g = Graph(direct=True, loops=True, multi=True)

    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(2, 2)
    g.add_edge(2, 3)

    for n in g.nodes:
        print(n, g.in_degree(n), g.out_degree(n))

    for e in g.edges:
        print(e[0], "->", e[1])


if __name__ == '__main__':
    main()
