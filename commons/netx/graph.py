from collections import defaultdict
from typing import Union, Iterator, Any

__version__ = "1.1.0"

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
from .edges import IEdges, DEdges, UEdges, DagEdges


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

        :param direct: if the graph id oriented
        :param acyclic: if the graph id oriented and acyclic
        :param loops: if the loops are supported
        :param multi: if there are multiple edges between 2 nodes
        :param gprops: graph properties
        """
        self._direct = direct or acyclic
        self._acyclic = acyclic
        self._props = gprops

        self._nodes: dict[int, dict] = {}

        if acyclic:
            self._edges: IEdges = DagEdges(multi)
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
        return self._direct

    def is_multigraph(self) -> bool:
        return self._edges.multi

    def has_loops(self) -> bool:
        return self._edges.loops

    def is_acyclic(self) -> bool:
        return self._acyclic

    # -----------------------------------------------------------------------
    # property G.nodes is NOT compatible with 'networkx'

    @property
    def nodes_(self) -> dict[int, dict]:
        return self._nodes

    # -----------------------------------------------------------------------
    # property G.edges is compatible with 'networkx'

    @property
    def edges(self) -> IEdges:
        return self._edges

    # -----------------------------------------------------------------------
    # For undirected graphs, succ and pred are the same

    @property
    def adj(self) -> dict[int, list[int]]:
        return self._edges.adj

    @property
    def succ(self) -> dict[int, list[int]]:
        return self._edges.succ

    @property
    def pred(self) -> dict[int, list[int]]:
        return self._edges.pred

    # -----------------------------------------------------------------------
    # Compatibility with 'networkx'

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
        return len(self._nodes)

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def nodes(self) -> Iterator[int]:
        return iter(self._nodes.keys())

    def has_node(self, n) -> bool:
        return n in self._nodes

    def add_node(self, n: int, **nprops):
        return self.add_node_(n, nprops)

    def add_node_(self, n: int, nprops: dict):
        if n not in self._nodes:
            self._nodes[n] = nprops
            self.adj[n] = []
        else:
            pass
        self.cache.clear()
        return self

    def add_nodes_from(self, nlist: list[int], **nprops):
        for n in nlist:
            if isinstance(n, tuple):
                v, vprops = n
                self.add_node_(v, nprops | vprops)
            else:
                self.add_node_(n, nprops)
        return self

    def neighbors(self, n: int, inbound=None) -> list[int]:
        """
        :param n:
        :param inbound:
            None    in/out
            False   out
            True    in
        :return:
        """
        return sorted(self._edges.neighbors(n, inbound))

    # compatibility
    def __contains__(self, n: int) -> bool:
        return n in self._nodes

    # -----------------------------------------------------------------------
    # Operations/edges
    #   G.edges
    #   e in G.edges

    def size(self) -> int:
        return len(self._edges)

    def number_of_edges(self) -> int:
        return len(self._edges)

    def has_edge(self, u: int, v: int) -> bool:
        return (u, v) in self._edges

    def add_edge(self, u: int, v: int, **eprops):
        return self.add_edge_(u, v, eprops)

    # edge props as dict
    def add_edge_(self, u: int, v: int, eprops: dict):
        self.add_node(u)
        self.add_node(v)
        self._edges.add_edge(u, v, eprops)
        self.cache.clear()
        return self

    # compatibility
    def add_edges_from(self,
                       elist: list[Union[tuple[int, int], tuple[int, int, dict]]],
                       **eprops):
        for e in elist:
            if len(e) == 2:
                u, v = e
                self.add_edge_(u, v, eprops)
            else:
                u, v, uvprops = e
                self.add_edge_(u, v, eprops | uvprops)
        return self

    # -----------------------------------------------------------------------
    # Node degree
    # Note: with multi=False, multiple edges are not counted
    #

    def degree(self, n: int, multi=False) -> int:
        if self._direct:
            return self.in_degree(n, multi=multi) + self.out_degree(n, multi=multi)
        else:
            return self.out_degree(n, multi=multi)

    def out_degree(self, u: int, multi=False) -> int:
        return self._edges.out_degree(u, multi=multi)

    def in_degree(self, v: int, multi=False) -> int:
        return self._edges.in_degree(v, multi=multi)

    # -----------------------------------------------------------------------
    #

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator:
        return iter(self._nodes)

    # -----------------------------------------------------------------------
    # Nodes or edges ???

    def __getitem__(self, n_e: Union[int, tuple[int, int]]) -> list[int]:
        """
        :param n_e: node or edge
        :return:
        """
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
        nv = len(self._nodes)
        ne = len(self._edges)
        return f"{self.name}=(|V|={nv}, |E|={ne})"
# end


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

