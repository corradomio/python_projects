import igraph as ig
from dotimport import DOTImporter
import matplotlib.pyplot as plt


class GraphImporter(DOTImporter):

    def __init__(self, g: ig.Graph, **kw):
        super().__init__(**kw)
        self.g = g

    def on_vertex(self, v, attributes):
        # v -= 1
        # self.g.add_vertex(v)
        self.g.add_vertex(str(v))
        pass

    def on_edge(self, v1, v2, directed):
        # v1 -= 1
        # v2 -= 1
        # self.g.add_edge(v1, v2)
        self.g.add_edge(str(v1), str(v2))
        pass


def read_file(file):
    with open(file, 'r') as file:
        return file.read()


def main():
    # data = read_file("dependency-tomcat.dot")
    data = read_file("dependency-hibernate.dot")
    g = ig.Graph(directed=True)
    importer = GraphImporter(g)
    importer.parse(data)

    plt.hist(g.degree(), bins=350)
    plt.show()
    print(g.maxdegree())

    # layout = g.layout("drl")
    # layout = g.layout("lgl")
    # ig.plot(g, layout=layout)
    # plt.show()


if __name__ == "__main__":
    main()
