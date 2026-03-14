import numpy as np
import cdt
import networkx as nx
from cdt.metrics import SID as CDT_SID

cdt.SETTINGS.rpath = r"D:\R\R-4.5.2-SID\bin\Rscript.exe"


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


def GADJID_SID(G1, G2):
    from gadjid import sid,  ancestor_aid, oset_aid, parent_aid, shd
    ROW_TO_COL = "from row to column"

    G1m = nx.adjacency_matrix(G1).todense().astype(np.int8)
    G2m = nx.adjacency_matrix(G2).todense().astype(np.int8)

    print("ancestor_aid:", ancestor_aid(G1m, G2m, edge_direction=ROW_TO_COL))
    print("   oset_aid:", oset_aid(G1m, G2m, edge_direction=ROW_TO_COL))
    print(" parent_aid:", parent_aid(G1m, G2m, edge_direction=ROW_TO_COL))
    print("        shd:", shd(G1m, G2m))
    print("        sid:", sid(G1m, G2m, edge_direction=ROW_TO_COL))

    # return sid(G1m, G2m, edge_direction=ROW_TO_COL)
    return "---"

def main():
    # print(CDT_SID(G, G))
    # print(CDT_SID(G, H1))
    # print(CDT_SID(G, H2))
    # print(CDT_SID(H1, H2))
    print("---")
    print(GADJID_SID(G, G))
    print(GADJID_SID(G, H1))
    print(GADJID_SID(G, H2))
    print(GADJID_SID(H1, H2))
# end


if __name__ == "__main__":
    main()
