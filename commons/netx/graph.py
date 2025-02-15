from collections import defaultdict
from typing import Union, Iterator, Any, Optional

__version__ = "1.1.0"

import numpy as np
from networkx.classes import is_directed

#
# Networkx graph types
#
#                   Type            Self-loops allowed      Parallel edges allowed
#   Graph           undirected      yes                     no
#   DiGraph         directed        yes                     no
#   MultiGraph      undirected      yes                     yes
#   MultiDiGraph    directed        yes                     yes
#
#
# Graph supported in this library:
#
#                   directed        Self-loops allowed      Parallel edges allowed      Cycles allowed
#   Graph           yes/no          yes/no                  yes/no                      yes/no
#

#
# Networkx compatibility
#
#   G=netx.Graph(...)
#   G.add_node(n ,{..})
#   G.add_nodes_from([n1, (n2, {..}), ...])             no
#   G.remove_node(n)
#   G.remove_nodes_from([n1, ...])
#   G.add_edge(u,v ,{..})
#   G.add_edges_from([[u1, v1], [u2, v2, {..}], ...])   no
#   G.clear()                                           no
#   G.number_of_nodes()
#   G.number_of_edges()
#   G.adj[v]
#   G.neighbors(v)
#   G.successors
#   G.predecessors
#   G.nodes
#   G.edges
#   .
from .types import NODE_TYPE, EDGE_TYPE
from .edges import IEdges, DEdges, UEdges, DAGEdges


# Networkx
#
#   G[u][v] returns the edge attribute dictionary.
#   G[u, v] as G[u][v]  (not supported)
#   n in G tests if node n is in graph G.
#   for n in G: iterates through the graph.
#   for nbr in G[n]: iterates through neighbors.
#   for e in list(G.edges): iterates through edges
#   for v in G.adj[u] | for v in G.succ[u] | for v in G.prec[u]
#

class Graph:

    def __init__(self,
                 direct=False,
                 loops=False,
                 multi=False,
                 acyclic=False,
                 **gprops):
        """

        :param direct: if the graph id oriented/directed
        :param acyclic: if the graph id oriented/directed and acyclic
        :param loops: if the loops are permitted
        :param multi: if multiple edges are permitted between 2 nodes
        :param gprops: graph properties
        """
        self._direct = direct or acyclic
        self._acyclic = acyclic
        self._props = gprops

        self._nodes: dict[int, dict] = {}

        if acyclic:
            self._edges: IEdges = DAGEdges(multi)
        elif direct:
            self._edges: IEdges = DEdges(loops, multi)
        else:
            self._edges: IEdges = UEdges(loops, multi)

        if "name" not in self._props:
            self._props["name"] = "G"

        # local cache used to save results of some complex computation
        # Cleared on EACH change
        self.cache: dict[str, Any] = defaultdict(lambda : {})
    # end

    def copy(self):
        # networks
        C = Graph(direct=self._direct,
                  loops=self.has_loops(),
                  multi=self.is_multigraph(),
                  acyclic=self._acyclic,
                  **self._props)

        C.add_nodes_from(self.nodes())
        C.add_edges_from(self.edges)
        return C


    # -----------------------------------------------------------------------
    # Properties

    @property
    def name(self) -> str:
        return self._props["name"]

    @property
    def properties(self, type=False) -> dict:
        if type:
            return self._props | dict(
                direct=self._direct,
                acyclic=self._acyclic,
                multi=self._edges.multi,
                loops=self._edges.loops
            )
        else:
            return self._props

    def is_directed(self) -> bool:
        # networkx
        return self._direct

    def is_multigraph(self) -> bool:
        # networkx
        return self._edges.multi

    def has_loops(self) -> bool:
        return self._edges.loops

    def is_acyclic(self) -> bool:
        return self._acyclic

    def is_dag(self) -> bool:
        return self._direct and self._acyclic

    # -----------------------------------------------------------------------
    # property G.nodes is NOT compatible with 'networkx'

    @property
    def nodes_(self) -> dict[NODE_TYPE, dict]:
        return self._nodes

    # -----------------------------------------------------------------------
    # property G.edges is compatible with 'networkx'

    @property
    def edges(self) -> IEdges:
        return self._edges

    # -----------------------------------------------------------------------
    # For undirected graphs, succ and pred are the same

    @property
    def adj(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._edges.adj

    @property
    def succ(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._edges.succ

    @property
    def pred(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._edges.pred

    # -----------------------------------------------------------------------
    # Compatibility with 'networkx' !

    @property
    def _succ(self):
        return self._edges.succ

    @property
    def _pred(self):
        return self._edges.pred

    # -----------------------------------------------------------------------
    # Operations/nodes
    #   G.nodes[n]
    #   n in G.nodes

    def order(self) -> int:
        # networkx
        return len(self._nodes)

    def number_of_nodes(self) -> int:
        # networkx
        return len(self._nodes)

    def nodes(self) -> Iterator[NODE_TYPE]:
        return iter(self._nodes.keys())

    def has_node(self, n: NODE_TYPE) -> bool:
        # networkx
        return n in self._nodes

    def add_node(self, n: NODE_TYPE, **nprops):
        # networkx
        return self.add_node_(n, nprops)

    def add_node_(self, n: NODE_TYPE, nprops: dict):
        if n not in self._nodes:
            self._nodes[n] = nprops
            self.adj[n] = []
        else:
            pass
        self.cache.clear()
        return self

    def add_nodes_from(self, nlist: list[NODE_TYPE] | Iterator[NODE_TYPE], **nprops):
        # networkx
        for n in nlist:
            if isinstance(n, tuple):
                v, vprops = n
                self.add_node_(v, nprops | vprops)
            else:
                self.add_node_(n, nprops)
        return self

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool] = None) -> list[NODE_TYPE]:
        """
        :param n:
        :param inbound:
            None    in/out
            False   out
            True    in
        :return:
        """
        # networkx
        return sorted(self._edges.neighbors(n, inbound))

    def node_edges(self, n: NODE_TYPE, inbound: Optional[bool] = None) -> dict[EDGE_TYPE, dict]:
        nnodes = self.neighbors(n, inbound=inbound)
        nedges: dict[EDGE_TYPE, dict] = {}
        if inbound:
            for u in nnodes:
                nedges[(u, n)] = self.edges[(u, n)]
        else:
            for v in nnodes:
                nedges[(n, v)] = self.edges[(n, v)]
        return nedges
    # end

    # -----------------------------------------------------------------------
    # compatibility with networkx
    # -----------------------------------------------------------------------

    def __contains__(self, n: int) -> bool:
        return n in self._nodes

    def remove_node(self, n: NODE_TYPE):
        if n not in self._nodes:
            return
        del self._nodes[n]
        self._edges.remove_node(n)

    def remove_nodes_from(self, nlist: list[NODE_TYPE]):
        for n in nlist:
            self.remove_node(n)

    def nbunch_iter(self, source) -> Iterator:
        if source is None:
            return self.nodes()
        if isinstance(source, list | tuple):
            for n in source:
                yield n
        else:
            yield source

    # -----------------------------------------------------------------------
    # Operations/edges
    #   G.edges
    #   e in G.edges

    def size(self) -> int:
        # networkx
        return len(self._edges)

    def number_of_edges(self) -> int:
        # networkx
        return len(self._edges)

    # def edges(self):
    #     return self._edges.keys()

    def has_edge(self, u: NODE_TYPE, v: NODE_TYPE) -> bool:
        # networkx
        return (u, v) in self._edges

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, **eprops):
        # networkx
        return self.add_edge_(u, v, eprops)

    # edge props as dict
    def add_edge_(self, u: NODE_TYPE, v: NODE_TYPE, eprops: dict):
        self.add_node(u)
        self.add_node(v)
        self._edges.add_edge(u, v, eprops)
        self.cache.clear()
        return self

    # compatibility
    def add_edges_from(self,
                       elist: list[Union[tuple[NODE_TYPE, NODE_TYPE], tuple[NODE_TYPE, NODE_TYPE, dict]]],
                       **eprops):
        # networkx
        for e in elist:
            if len(e) == 2:
                u, v = e
                self.add_edge_(u, v, eprops)
            else:
                u, v, uvprops = e
                self.add_edge_(u, v, eprops | uvprops)
        return self

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        # networkx
        self._edges.remove_edge(u, v)

    def remove_edges_from(self, elist: list[EDGE_TYPE]):
        # networkx
        for e in elist:
            self.remove_edge(*e)

    def remove_singletons(self):
        singletons = [n for n in self if self.degree(n) == 0]
        self.remove_nodes_from(singletons)

    # -----------------------------------------------------------------------
    # Node degree
    # Note: with multi=False, multiple edges are not counted
    #

    def degree(self, n: NODE_TYPE, multi=False) -> int:
        if self._direct:
            return self.in_degree(n, multi=multi) + self.out_degree(n, multi=multi)
        else:
            return self.out_degree(n, multi=multi)

    def out_degree(self, u: NODE_TYPE, multi=False) -> int:
        return self._edges.out_degree(u, multi=multi)

    def in_degree(self, v: NODE_TYPE, multi=False) -> int:
        return self._edges.in_degree(v, multi=multi)

    # -----------------------------------------------------------------------
    #

    def __len__(self) -> int:
        # networkx
        return len(self._nodes)

    def __iter__(self) -> Iterator:
        # networkx
        return iter(self._nodes)

    # -----------------------------------------------------------------------
    # Nodes or edges ???

    def __getitem__(self, n_e: Union[int, tuple[int, int]]) -> list[int]:
        """
        :param n_e: node or edge
        :return:
        """
        # networkx
        return self._edges.succ[n_e]

    # -----------------------------------------------------------------------

    def clone(self, name=None):
        G = Graph(
            name=name,
            direct=self.is_directed(),
            loops=self.has_loops(),
            multi=self.is_multigraph(),
            acyclic=self.is_acyclic()
        )

        for n in self._nodes:
            G.add_node_(n, self._nodes[n])

        for uv in self._edges:
            u, v = uv
            G.add_edge_(u, v, self._edges[uv])

        return G

    # -----------------------------------------------------------------------

    def dump(self):
        print(f"{self.name}: |V|={self.order()}, |E|={self.size()}")
        print(f"    directed: {self.is_directed()}")
        print(f"       loops: {self.has_loops()}")
        print(f"     acyclic: {self.is_acyclic()}")
        print(f"  multigraph: {self.is_multigraph()}")

    # -----------------------------------------------------------------------

    def __repr__(self):
        # networkx
        nv = len(self._nodes)
        ne = len(self._edges)
        return f"{self.name}=(|V|={nv}, |E|={ne})"

    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# Adjacency matrix
# ---------------------------------------------------------------------------

def adjacency_matrix(G: Graph) -> np.ndarray:
    n = G.order()
    is_directed = G.is_directed()
    A = np.zeros((n, n), dtype=int)
    for e in G.edges:
        u, v = e
        w = 1
        if is_directed:
            A[u, v] = w
        else:
            A[u, v] = w
            A[v, u] = w
    # end
    return A


# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------

class DiGraph(Graph):
    def __init__(self, loops=False, **gprops):
        super().__init__(direct=True, loops=loops, **gprops)


class DirectAcyclicGraph(Graph):
    def __init__(self, **gprops):
        super().__init__(acyclic=True, direct=True, loops=False, **gprops)


class MultiGraph(Graph):
    def __init__(self, loops=False, **gprops):
        super().__init__(multi=True, loops=loops, **gprops)


class MultiDiGraph(Graph):
    def __init__(self, loops=False, **gprops):
        super().__init__(direct=True, multi=True, loops=loops, **gprops)


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

