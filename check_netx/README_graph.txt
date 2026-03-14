G: Graph
    number_of_nodes(), order()
    add_node(n, {nprops})
    add_nodes_from([n,, ...], {nprops})
    add_nodes_from([(, {nprops},, ...])
    add_nodes_from(H)
    add_node(H)

    number_of_edges(), size()

    add_edge(u,v, {epops})
    add_edges_from([(u,v, {eprops}), ...], {eprops})
    add_edges_from(H.edges)

    adj[u][v] -> {nprops}
    successors(u) -> [v1,...]
    predecessors(v) -> [u1, ...]

    nodes | nodes() | nodes[u][attr]
    edges | edges() | edges([u1,...]) -> EdgeDataView | edges[(u,v)][attr]
    adj[u] | neighbours(u)
    degree[u] | degree(u) | degree([u1,...]) -> DegreeView
    items()
    data()

    G[u] == G.adj[u]
    G[u][v] == G.edges[u,v] -> eprops
    G.adjacency() == G.adj.items()
    G.graph -> gprops

G: DiGraph
    out_edges
    in_degree
    predecessors() succesors()
    neighbours == succesors()
    degree = in_degree+out_degree