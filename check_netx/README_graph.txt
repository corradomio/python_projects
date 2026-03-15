G: Graph
    _node
    _adj

    property nodes      -> NodeView
    property adj        -> AdjacencyView
    property edges      -> EdgeView
    property degrees    -> DegreeView

   -number_of_nodes(),
   -order()
   -len(G)
   -add_node(n, {nprops})
   -add_nodes_from([n,, ...], {nprops})
   -add_nodes_from([(, {nprops},, ...])
    add_nodes_from(H)

   -number_of_edges(),
   -size()
   -add_edge(u,v, {epops})
   -add_edges_from([(u,v, {eprops}), ...], {eprops})
    add_edges_from(H.edges)

    adj[u][v] -> {nprops}
   -successors(u) -> [v1,...]
   -predecessors(v) -> [u1, ...]

    nodes | nodes()
    nodes[u][attr]
    edges | edges()
    edges([u1,...]) -> EdgeDataView
    edges[(u,v)][attr]
    adj[u] |
   -neighbours(u)   -> key_dict_iterator
   -(in_|out_|)degree[u] | (in_|out_|)degree(u)
    degree([u1,...]) -> DegreeView
    items()
    data()

    G[u] == G.adj[u]
    G[u][v] == G.edges[u,v] -> eprops
    G.adjacency() == G.adj.items()
    G.graph -> gprops

G: DiGraph
    _node
    _adj
    _prec
    _succ

    property adj    -> AdjacencyView
    property prec   -> AdjacencyView
    property succ   -> AdjacencyView

    property edges      -> OutEdgesView
    property out_edges  -> OutEdgesView
    property in_edges   -> InEdgesView

    property degree     -> DiDegreeView (out_degree + in_degree)
    property out_degree -> OutDegreeView
    property in_degree  -> InDegreeView

    out_edges           -> OutEdgeView
    in_degree           -> InEdgesView
    predecessors(n)     -> key_dict_iterator
    successors(n)       -> key_dict_iterator
    neighbours(n) == successors(n)

    has_successor(u, v)
    has_predecessor(u, v)


G.adj -> AdjacencyView
G.adj[n] -> AtlasView