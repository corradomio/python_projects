from collections import defaultdict
from typing import Union, Iterator, Any, Optional
from functools import cached_property
from collections import defaultdict, deque
from typing import Optional

from .types import NODE_TYPE, EDGE_TYPE

__version__ = "1.1.1"

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

# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

class IEdges(dict):
    """
    Dictionary
        (u, v) -> { ... }
    """

    def __init__(self, loops: bool, multi: bool):
        super().__init__()
        self.loops = loops
        self.multi = multi

    @property
    def adj(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        """
        Adjacency list:
            u -> [v1, v2, ...]
        """
        return {}

    @property
    def succ(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        """
        Successors of a node
        """
        return {}

    @property
    def pred(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        """
        Predecessors of a node
        :return:
        """
        return {}

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool]) -> list[NODE_TYPE]:
        """
        Neighbors of a node
        :param n: node
        :param inbound: if to consider the inbound edges only
        :return:
        """
        return []

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops):
        """
        Add an edge.
        This is ONLY the second step.
        The first one is implemented in the derived classes

        :param u:
        :param v:
        :param eprops:
        :return:
        """
        # check for loop, multi, already present
        if not self._check_edge(u, v):
            return self

        if v not in self.adj[u]:
            self[(u, v)] = [eprops] if self.multi else eprops
        elif self.multi:
            self[(u, v)].append(eprops)

        return self
    # end

    def _check_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        # check if u == v (loop)
        # check if (u,v) is an edge already present
        #   (multiple edges)
        # check for dag
        if not self.loops and u == v:
            return False
        if not self.multi and (u, v) in self:
            return False
        return True

    def out_degree(self, u: NODE_TYPE, multi=False) -> int:
        if not self.multi or not multi:
            return len(self.succ[u])
        else:
            deg = 0
            for v in self.succ[u]:
                deg += len(self[(u, v)])
            return deg

    def in_degree(self, v: NODE_TYPE, multi=False) -> int:
        if not self.multi or not multi:
            return len(self.pred[v])
        else:
            deg = 0
            for u in self.pred[v]:
                deg += len(self[(u, v)])
            return deg

    # def __getitem__(self, uv: tuple[int, int]) -> list[int]: return []
    # def __setitem__(self, uv: tuple[int, int], value: list[int]): ...
    # def __len__(self): return 0

    def remove_node(self, n: NODE_TYPE):
        elist = []
        for e in self:
            u, v = e
            if u == n or v == n:
                elist.append(e)

        for e in elist:
            del self[e]

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        e = (u, v)
        if e in self:
            del self[e]
# end


class UEdges(IEdges):
    """Undirected edges dictionary"""

    def __init__(self, loops: bool, multi: bool):
        super().__init__(loops, multi)
        self._adj: dict[NODE_TYPE, list[NODE_TYPE]] = defaultdict(lambda: list())

    @property
    def adj(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._adj

    @property
    def succ(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._adj

    @property
    def pred(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._adj

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops):
        if u > v:
            u, v = v, u
        super().add_edge(u, v, eprops)

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool]) -> list[int]:
        return self._adj[n]

    def remove_node(self, n: NODE_TYPE):
        del self._adj[n]
        super().remove_node(n)

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        if u > v:
            u, v = v, u
        super().remove_edge(u, v)

    def __contains__(self, uv):
        u, v = uv
        if u > v:
            uv = v, u
        return super().__contains__(uv)

    def __getitem__(self, uv):
        u, v = uv
        if u > v:
            uv = v, u
        return super().__getitem__(uv)

    def __setitem__(self, uv, eprops):
        u, v = uv
        if u > v:
            uv = v, u
        if not self.loops and u == v:
            return None
        elif super().__contains__(uv):
            return super().__setitem__(uv, eprops)
        else:
            if u not in self._adj or v not in self._adj[u]:
                self._adj[u].append(v)
                self._adj[v].append(u)
            # end
            return super().__setitem__(uv, eprops)
    # end
# end


class DEdges(IEdges):
    """Directed edges dictionary"""

    def __init__(self, loops: bool, multi: bool):
        super().__init__(loops, multi)

        self._succ: dict[NODE_TYPE, list[NODE_TYPE]] = defaultdict(lambda: list())
        self._prec: dict[NODE_TYPE, list[NODE_TYPE]] = defaultdict(lambda: list())

    @property
    def adj(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._succ

    @property
    def succ(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._succ

    @property
    def pred(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        return self._prec

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops: dict):
        super().add_edge(u, v, eprops)

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool]) -> list[NODE_TYPE]:
        if inbound is None:
            return self._prec[n] + self._succ[n]
        elif inbound:
            return self._prec[n]
        else:
            return self._succ[n]

    def remove_node(self, n: NODE_TYPE):
        if n in self._prec[n]:
            del self._prec[n]
        if n in self._succ[n]:
            del self._succ[n]
        super().remove_node(n)

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        super().remove_edge(u, v)

    def __contains__(self, uv):
        return super().__contains__(uv)

    def __getitem__(self, uv):
        return super().__getitem__(uv)

    def __setitem__(self, uv, epros):
        u, v = uv
        if not self.loops and u == v:
            return None
        elif super().__contains__(uv):
            return super().__setitem__(uv, epros)
        else:
            self._succ[u].append(v)
            self._prec[v].append(u)
            return super().__setitem__(uv, epros)
    # end
# end


class DAGEdges(DEdges):
    def __init__(self, multi: bool):
        super().__init__(False, multi)

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops):
        if self.has_path(v, u):
            return
        else:
            super().add_edge(u, v, eprops)

    def has_path(self, u: NODE_TYPE, v: NODE_TYPE) -> bool:
        processed = set()
        toprocess = deque([u])
        while toprocess:
            t = toprocess.popleft()
            if t == v:
                return True
            elif t in processed:
                continue
            toprocess.extend(self.succ[t])
            processed.add(t)
        return False
# end


# ---------------------------------------------------------------------------
# Graph
#   DiGraph
# ---------------------------------------------------------------------------

class DegreeView:
    IN_DEGREE = 1
    OUT_DEGREE = 2
    DEGREE = 3

    def __init__(self, degree_type: int, G, multi: bool):
        self.degree_type = degree_type
        self.G = G
        self.multi = multi

    def __getitem__(self, n: NODE_TYPE):
        if self.degree_type == self.IN_DEGREE:
            return self.G._edges.in_degree(n, multi=self.multi)
        if self.degree_type == self.OUT_DEGREE:
            return self.G._edges.out_degree(n, multi=self.multi)
        if self.G._direct:
            return self.G._edges.in_degree(n, multi=self.multi) + self.G._edges.out_degree(n, multi=self.multi)
        else:
            return self.G._edges.out_degree(n, multi=self.multi)
# end


class EdgesView:
    IN_EDGES = 1
    OUT_EDGES = 2
    EDGES = 3

    def __init__(self, edge_type: int, G):
        self.edge_type = edge_type
        self.G = G

    def __call__(self, *args, **kwargs):
        return self.G._edges.keys()

    def __getitem__(self, n: NODE_TYPE):
        if self.edge_type == EdgesView.IN_EDGES:
            return self.G._edges.pred[n]
        if self.edge_type == EdgesView.OUT_EDGES:
            return self.G._edges.succ[n]
        else:
            return self.G._edges.adj[n]

    def __iter__(self):
        if self.edge_type == EdgesView.IN_EDGES:
            return iter(self.G._edges.pred)
        if self.edge_type == EdgesView.OUT_EDGES:
            return iter(self.G._edges.succ)
        else:
            return iter(self.G._edges.adj)
# end


class NodesView:
    def __init__(self, G):
        self.G = G

    def __call__(self, *args, **kwargs):
        return iter(self.G._nodes)

    def __iter__(self):
        return iter(self.G._nodes)

    def __contains__(self, n: NODE_TYPE):
        return n in self.G._nodes

    def __getitem__(self, n: NODE_TYPE):
        return self.G._nodes[n]


class AdjacencyView:
    def __init__(self, G, adjacency_dict: dict):
        self.G = G
        self.adjacency_dict = adjacency_dict

    def __getitem__(self, n: NODE_TYPE):
        return self.adjacency_dict[n]
# end


# ---------------------------------------------------------------------------
# Graph
#   DiGraph
# ---------------------------------------------------------------------------

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
        C.add_edges_from(self.edges())
        return C


    # -----------------------------------------------------------------------
    # Properties

    @property
    def name(self) -> str:
        # networkx
        return self._props["name"]

    @property
    def properties(self, type=False) -> dict:
        """
        Graph properties

        :param type: if to add the graph's type properties
                     (direct, acyclic, multi, loops)
        :return:
        """
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
    # Nodes
    # -----------------------------------------------------------------------

    def __iter__(self) -> Iterator[NODE_TYPE]:
        # networkx
        return iter(self._nodes)

    def __contains__(self, n: NODE_TYPE) -> bool:
        # networkx
        return n in self._nodes

    def __len__(self) -> int:
        # networkx
        return len(self._nodes)

    def __getitem__(self, n: NODE_TYPE) -> dict:
        # networkx
        return self._nodes[n]

    # -----------------------------------------------------------------------

    def order(self) -> int:
        # networkx
        return len(self._nodes)

    def number_of_nodes(self) -> int:
        # networkx
        return len(self._nodes)

    # -----------------------------------------------------------------------

    @cached_property
    def nodes(self):
        # networkx
        # return self._nodes.keys()
        return NodesView(self)

    def add_node(self, n: NODE_TYPE, **nprops):
        # networkx
        return self._add_node(n, nprops)

    def add_nodes_from(self, nlist: list[NODE_TYPE] | Iterator[NODE_TYPE], **nprops):
        # networkx
        for n in nlist:
            if isinstance(n, tuple):
                v, vprops = n
                self._add_node(v, nprops | vprops)
            else:
                self._add_node(n, nprops)
        return self

    def remove_node(self, n: NODE_TYPE):
        # networkx
        if n not in self._nodes:
            return
        del self._nodes[n]
        self._edges.remove_node(n)

    def remove_nodes_from(self, nlist: list[NODE_TYPE]):
        # networkx
        for n in nlist:
            self.remove_node(n)

    def has_node(self, n: NODE_TYPE) -> bool:
        # networkx
        return n in self._nodes

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

    def nbunch_iter(self, source: Optional[list|tuple|NODE_TYPE]) -> Iterator[NODE_TYPE]:
        # networkx
        if source is None:
            return self.nodes()
        if isinstance(source, list | tuple):
            for n in source:
                yield n
        else:
            yield source

    # -----------------------------------------------------------------------

    # def node_edges(self, n: NODE_TYPE, inbound: Optional[bool] = None) -> dict[EDGE_TYPE, dict]:
    #     nnodes = self.neighbors(n, inbound=inbound)
    #     nedges: dict[EDGE_TYPE, dict] = {}
    #     if inbound:
    #         for u in nnodes:
    #             nedges[(u, n)] = self._edges[(u, n)]
    #     else:
    #         for v in nnodes:
    #             nedges[(n, v)] = self._edges[(n, v)]
    #     return nedges

    def _add_node(self, n: NODE_TYPE, nprops: dict):
        if n not in self._nodes:
            self._nodes[n] = nprops
            self.adj[n] = []
        else:
            pass
        self.cache.clear()
        return self

    # -----------------------------------------------------------------------
    # Operations/edges
    #   G.edges
    #   e in G.edges

    @cached_property
    def edges(self):
        # networkx
        # return self._edges.keys()
        return EdgesView(EdgesView.EDGES, self)

    def in_edges(self):
        # networkx (DiGraph)
        return EdgesView(EdgesView.IN_EDGES, self)

    def out_edges(self):
        # networkx (DiGraph)
        return EdgesView(EdgesView.OUT_EDGES, self)

    def size(self) -> int:
        # networkx
        return len(self._edges)

    def number_of_edges(self) -> int:
        # networkx
        return len(self._edges)

    # -----------------------------------------------------------------------

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, **eprops):
        # networkx
        return self._add_edge(u, v, eprops)

    def add_edges_from(self, elist: list[EDGE_TYPE] | Iterator[EDGE_TYPE], **eprops):
        # networkx
        for e in elist:
            if len(e) == 2:
                u, v = e
                self._add_edge(u, v, eprops)
            else:
                u, v, uvprops = e
                self._add_edge(u, v, eprops | uvprops)
        return self

    # def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        # networkx
        self._edges.remove_edge(u, v)

    def remove_edges_from(self, elist: list[EDGE_TYPE]):
        # networkx
        for e in elist:
            self.remove_edge(*e)

    # def update(self, edges=None, nodes=None):

    def has_edge(self, u: NODE_TYPE, v: NODE_TYPE) -> bool:
        # networkx
        return (u, v) in self._edges

    # def get_edge_data(self, u, v, default=None):

    # def adjacency(self):

    # -----------------------------------------------------------------------

    # def clear(self):

    # def clear_edges(self):

    # -----------------------------------------------------------------------

    def remove_singletons(self):
        singletons = [n for n in self if self.degree[n] == 0]
        self.remove_nodes_from(singletons)

    # -----------------------------------------------------------------------

    # edge props as dict
    def _add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops: dict):
        self.add_node(u)
        self.add_node(v)
        self._edges.add_edge(u, v, eprops)
        self.cache.clear()
        return self

    # -----------------------------------------------------------------------
    # For undirected graphs, succ and pred are the same

    @property
    def adj(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        # networkx
        return self._edges.adj

    @property
    def succ(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        # networkx (DiGraph)
        return self._edges.succ

    @property
    def pred(self) -> dict[NODE_TYPE, list[NODE_TYPE]]:
        # networkx (DiGraph)
        return self._edges.pred

    # -----------------------------------------------------------------------
    # DiGraph

    # def has_successor(self, u, v):
    # def has_predecessor(self, u, v):
    # def successors(self, n):
    # def predecessors(self, n):
    # neighbors = successors

    # -----------------------------------------------------------------------
    # Node degree
    # Note: with multi=False, multiple edges are not counted
    #

    @property
    def degree(self, multi=False):
        # networkx
        return DegreeView(DegreeView.DEGREE, self, multi)

    @property
    def in_degree(self, multi=False):
        # networkx (DiGraph)
        return DegreeView(DegreeView.IN_DEGREE, self, multi)

    @property
    def out_degree(self, multi=False):
        # networkx (DiGraph)
        return DegreeView(DegreeView.OUT_DEGREE, self, multi)

    # -----------------------------------------------------------------------

    def clone(self, name=None):
        # networkx
        G = Graph(
            name=name,
            direct=self.is_directed(),
            loops=self.has_loops(),
            multi=self.is_multigraph(),
            acyclic=self.is_acyclic()
        )

        for n in self._nodes:
            G._add_node(n, self._nodes[n])

        for uv in self._edges:
            u, v = uv
            G._add_edge(u, v, self._edges[uv])

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

