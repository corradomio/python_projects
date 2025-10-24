import netx
import netx.util
import networkx as nx


def check_digraph_paths():
    G = netx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)])

    idx = 0
    for u in G.nodes:
        for v in G.nodes:
            if u == v: continue
            for P in netx.all_simple_paths(G, u, v):
                idx += 1
                print(f"[{idx:2}] {u}->{v}: {P}")

    print("--- ")
    G = nx.DiGraph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)])

    idx = 0
    for u in G.nodes:
        for v in G.nodes:
            if u == v: continue
            for P in nx.all_simple_paths(G, u, v):
                idx += 1
                print(f"[{idx:2}] {u}->{v}: {P}")


def check_graph_paths():
    G = netx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)])

    idx = 0
    for u in G.nodes:
        for v in G.nodes:
            if u == v: continue
            for P in netx.all_simple_paths(G, u, v):
                idx += 1
                print(f"[{idx:2}] {u}->{v}: {P}")

    print("--- ")
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)])

    idx = 0
    for u in G.nodes:
        for v in G.nodes:
            if u == v: continue
            for P in nx.all_simple_paths(G, u, v):
                idx += 1
                print(f"[{idx:2}] {u}->{v}: {P}")


def main():
    # check_digraph_paths()
    check_graph_paths()


if __name__ == "__main__":
    main()
