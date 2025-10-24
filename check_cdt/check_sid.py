import networkx as nx
from cdt.metrics import SID

G = nx.DiGraph()
G.add_edges_from([
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])

H1 = nx.DiGraph()
H1.add_edges_from([
    ["X1", "X2"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"],
    ["Y1","Y2"]
])

H2 = nx.DiGraph()
H2.add_edges_from([
    ["X2", "X1"],
    ["X1", "Y1"], ["X1", "Y2"], ["X1", "Y3"],
    ["X2", "Y1"], ["X2", "Y2"], ["X2", "Y3"]
])

H = nx.DiGraph()
H.add_edges_from([
    # ["Q","X","B","Y"],
    ["Q","X"],["X","B"],["B","Y"],
    # ["P","X","A"],
    ["P","X"],["X","A"],
    ["P","Y",],
    ["B","F"]
])



def main():
    print(SID(G, G))
    print(SID(G, H1))
    print(SID(G, H2))



if __name__ == "__main__":
    main()
