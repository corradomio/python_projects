import networkx as nx
import pandas as pd


def main():
    print(int('770471'))

    c: pd.DataFrame = pd.read_csv("4c00c2ca-type-component-vertices-1-r00-classified.csv")
    print(c.head())
    print(c.columns)
    print(len(c))
    g = nx.drawing.nx_pydot.read_dot("4c00c2ca-component-graph-1-r00.dot")
    print(g)
    for n in g.nodes():
        id = g.nodes[n]['typeId']
        print(id)
        cat = c[c["id"] == int(id)]["category"]
        print(id, "->", cat)
    # end
# end


if __name__ == "__main__":
    main()
