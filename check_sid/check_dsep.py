import sys
import netx
import netx as nx
from itertoolsx import subsets

sys.stdout = open("dsep_nx2.txt", mode="w")

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
    ["Q","X"],["X","B"],["B","Y"],
    ["P","X"],["X","A"],
    ["P","Y",],
    ["B","F"]
])


def ls(S): return list(sorted(S))

def ssorted(N):
    ss = subsets(N)
    ss = map(sorted, ss)
    ss = map(tuple, ss)
    ss = list(ss)
    ss = sorted(ss)
    ss = list(ss)
    return ss


def main():
    V = ls(H.nodes)
    for u in V:
        for v in V:
            if u == v: continue
            N = ls(set(V).difference([u,v]))
            ZZ = ssorted(N)
            for Z in ZZ:
                print(f"{u}->{v} | {Z}", netx.is_d_separator(H, u, v, set(Z)))
    pass


if __name__ == "__main__":
    main()
