import netx
import netx.util


# G = netx.DiGraph()
# G.add_edges_from([(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)])

def main():
    # G = netx.DiGraph()
    # G.add_path_from(["x","r","s","t"])
    # G.add_path_from(["v","u","t"])
    # G.add_path_from(["v","y"])
    # N = set(G.nodes)
    #
    # netx.util.draw(G)
    # netx.util.show()
    #
    # print(netx.all_chains(G, N))
    # print(netx.all_forks(G, N))
    # print(netx.all_colliders(G, N))
    #
    # N = set(G.nodes)
    #
    # for u in G.nodes:
    #     for v in G.nodes:
    #         if u == v: continue
    #         for P in netx.all_simple_paths(G, u, v):
    #             # print(netx.all_chains(G,P,P))
    #             # print(netx.all_forks(G,P,P))
    #             # print(netx.all_colliders(G,P,P))
    #             print(P, netx.all_paths_blocked(G, u, v, P))

    G = netx.DiGraph()
    G.add_edges_from([("X", "Y"), ("X", "B"), ("Y", "C"), ("A", "X"), ("A", "Y"), ("B", "Y"), ("B", "C")])

    assert netx.all_paths_blocked(G, "X", "Y", ["A"])
    assert netx.all_paths_blocked(G, "X", "Y", ["A", "B", "C"])



if __name__ == "__main__":
    main()
