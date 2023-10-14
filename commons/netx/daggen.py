from random import randrange, choice

import networkx as nx


def random_dag(n, e, connected=True, seed=None):
    """
    Generate a random directed graph
    Args:
        n: n of nodes
        e: n of vertices
        seed: if specified

    Returns:

    """
    G = nx.DiGraph()
    G.add_nodes_from(list(range(n)))
    completed = False

    while not completed:
        u = randrange(n)
        v = randrange(n)
        if u == v or G.has_edge(u, v) or G.has_edge(v, u):
            continue
        else:
            G.add_edge(u, v)
        try:
            cycle = nx.find_cycle(G, orientation='original')
            r = choice(cycle)
            G.remove_edge(r[0], r[1])
        except nx.NetworkXNoCycle as ex:
            pass
        is_connected = nx.is_weakly_connected(G)

        if connected and not is_connected:
            continue
        if len(G.edges) < e:
            continue
        completed = True
    # end
    print(nx.is_weakly_connected(G))
    return G