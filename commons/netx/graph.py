
__all__ = [
    "Graph", "NetxGraph",
    "DiGraph", "NetxDiGraph",
    "DirectAcyclicGraph",
    "MultiGraph",
    "MultiDiGraph",

    "is_netx_graph",
    "is_networkx_graph",
    "create_like",
    "print_graph_stats"
]

import networkx as nx
from collections import defaultdict, deque
from functools import cached_property
from typing import Optional, Any
from typing import Union, Iterator, Collection

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

# A DirectedAcyclicGraph (DAG) is a directed graph without cycles
# A PartialDirectedGraph (PDG) is a directed graph where for some edge
#   it is not specified the direction

# ---------------------------------------------------------------------------
# _ukey
# ---------------------------------------------------------------------------

def _ukey(key):
    u, v = key
    return key if u < v else (v, u)

# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
# IEdges
#   it is a dictionary with key (u,v) and value the edge properties
#   it contains
#

class IEdges(dict):
    """
    Dictionary
        (u, v) -> { ... }
    """

    def __init__(self, loops: bool, multi: bool):
        super().__init__()
        self.directed = False
        self.loops = loops
        self.multi = multi

    @property
    def adj(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        """
        Adjacency list:
            u -> [v1, v2, ...]
        """
        return {}

    @property
    def succ(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        """
        Successors of a node
        """
        return {}

    @property
    def pred(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        """
        Predecessors of a node
        :return:
        """
        return {}

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool]) -> set[NODE_TYPE]:
        """
        Neighbors of a node
        :param n: node
        :param inbound: if to consider the inbound edges only
        :return:
        """
        return set()

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
        elif len(eprops) > 0 and not self.multi:
            self[(u, v)] = self[(u, v)] | eprops
        elif self.multi:
            self[(u, v)].append(eprops)

        return self
    # end

    def get_edge(self, u, v):
        return self.__getitem__((u, v))

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
        self._adj: dict[NODE_TYPE, set[NODE_TYPE]] = defaultdict(lambda: set())
        pass

    # -----------------------------------------------------------------------
    #
    # def get(self, key, default=None):
    #     return super().get(_ukey(key), default)
    #
    # def __getitem__(self, key):
    #     return super().__getitem__(_ukey(key))
    #
    # def __setitem__(self, key, value):
    #     return super().__setitem__(_ukey(key), value)
    #
    # def __contains__(self, key):
    #     return super().__contains__(_ukey(key))
    #
    # def __delitem__(self, key):
    #     return super().__delitem__(_ukey(key))
    #
    # -----------------------------------------------------------------------

    @property
    def adj(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        return self._adj

    @property
    def succ(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        return self._adj

    @property
    def pred(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        return self._adj

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops: dict):
        super().add_edge(u, v, eprops)

    def get_edge(self, u: NODE_TYPE, v: NODE_TYPE) -> dict:
        return self[(u, v)]

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool]) -> set[NODE_TYPE]:
        return self._adj[n]

    def remove_node(self, n: NODE_TYPE):
        del self._adj[n]
        super().remove_node(n)

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        super().remove_edge(u, v)

    def __contains__(self, uv):
        uv = _ukey(uv)
        return super().__contains__(uv)

    def __getitem__(self, uv):
        uv = _ukey(uv)
        return super().__getitem__(uv)

    def __setitem__(self, uv, eprops):
        u, v = _ukey(uv)
        if not self.loops and u == v:
            return None
        elif super().__contains__(uv):
            return super().__setitem__(uv, eprops)
        else:
            if u not in self._adj or v not in self._adj[u]:
                self._adj[u].add(v)
                self._adj[v].add(u)
            # end
            return super().__setitem__(uv, eprops)
    # end

    def __iter__(self):
        for u in self._adj:
            for v in self._adj[u]:
                yield u,v
# end


class DEdges(IEdges):
    """Directed edges dictionary"""

    def __init__(self, loops: bool, multi: bool):
        super().__init__(loops, multi)
        self.directed = True
        self._succ: dict[NODE_TYPE, set[NODE_TYPE]] = defaultdict(lambda: set())
        self._pred: dict[NODE_TYPE, set[NODE_TYPE]] = defaultdict(lambda: set())
        pass

    @property
    def adj(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        return self._succ

    @property
    def succ(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        return self._succ

    @property
    def pred(self) -> dict[NODE_TYPE, set[NODE_TYPE]]:
        return self._pred

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, eprops: dict):
        super().add_edge(u, v, eprops)

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool]) -> set[NODE_TYPE]:
        if inbound is None:
            return self._pred[n].union(self._succ[n])
        elif inbound:
            return self._pred[n]
        else:
            return self._succ[n]

    def remove_node(self, n: NODE_TYPE):
        if n in self._pred[n]:
            del self._pred[n]
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
            self._succ[u].add(v)
            self._pred[v].add(u)
            return super().__setitem__(uv, epros)
    # end

    def __iter__(self):
        for u in self._succ:
            for v in self._succ[u]:
                yield u,v
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
    DEGREE = 0

    def __init__(self, edges: IEdges, degree_type: int):
        self._edges = edges
        self._degree_type = degree_type
        self._directed = edges.directed
        self._multi = edges.multi

    def __call__(self, n: Optional[NODE_TYPE]=None):
        if n is None:
            return self
        else:
            return self._get_degree(n)

    def __getitem__(self, n: NODE_TYPE):
        return self._get_degree(n)

    def _get_degree(self, n: NODE_TYPE) -> int:
        if self._degree_type == self.IN_DEGREE:
            return self._edges.in_degree(n, multi=self._multi)
        if self._degree_type == self.OUT_DEGREE:
            return self._edges.out_degree(n, multi=self._multi)
        if self._directed:
            return self._edges.in_degree(n, multi=self._multi) + self._edges.out_degree(n, multi=self._multi)
        else:
            return self._edges.out_degree(n, multi=self._multi)
# end


class EdgesView:
    IN_EDGES = 1
    OUT_EDGES = 2
    EDGES = 0

    def __init__(self, edges, edge_type: int):
        self._edges = edges
        self._edge_type = edge_type

    def __call__(self, *args, **kwargs):
        return self._edges.keys()

    def __getitem__(self, n: NODE_TYPE):
        if isinstance(n, tuple):
            assert len(n) == 2
            u, v = n
            return self._edges.get_edge(u, v)
        if self._edge_type == EdgesView.IN_EDGES:
            return self._edges.pred[n]
        if self._edge_type == EdgesView.OUT_EDGES:
            return self._edges.succ[n]
        else:
            return self._edges.adj[n]

    def __iter__(self):
        # if self._edge_type == EdgesView.IN_EDGES:
        #     return iter(self._edges.__iter__())
        # if self._edge_type == EdgesView.OUT_EDGES:
        #     return iter(self._edges.__iter__())
        # else:
        #     return iter(self._edges.__iter__())
        return self._edges.__iter__()
# end


class NodesView:
    def __init__(self, nodes):
        self._nodes = nodes

    def __call__(self, *args, **kwargs):
        return iter(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __contains__(self, n: NODE_TYPE):
        return n in self._nodes

    def __getitem__(self, n: NODE_TYPE):
        return self._nodes[n]
# end


class AtlasView:

    def __init__(self, edges: IEdges, adj: dict, n: NODE_TYPE, rev: bool):
        assert isinstance(edges, IEdges)
        assert isinstance(adj, dict)
        self._edges = edges
        self._adj = adj
        self._n = n
        self._rev = rev

    def __call__(self, v: NODE_TYPE):
        return self.__getitem__(v)

    def __getitem__(self, v: NODE_TYPE):
        uv = (v,self._n) if self._rev else (self._n,v)
        return self._edges.get_edge(*uv)

    def __repr__(self):
        return f"AtlasView({self._n})"


class AdjacencyView:
    def __init__(self, edges: IEdges, adj: dict, rev=False):
        assert isinstance(edges, IEdges)
        assert isinstance(adj, dict)
        self._edges = edges
        self._adj = adj
        self._rev = rev

    def __call__(self, n: Optional[NODE_TYPE]=None):
        if n is None:
            return self
        else:
            return AtlasView(self._edges, self._adj, n, self._rev)

    def __getitem__(self, n: NODE_TYPE):
        return AtlasView(self._edges, self._adj, n, self._rev)
# end


# ---------------------------------------------------------------------------
# Graph
#   DiGraph
# ---------------------------------------------------------------------------

class Graph:

    def __init__(self,
                 directed=False,
                 loops=False,
                 multi=False,
                 acyclic=False,
                 **gprops):
        """

        :param directed: if the graph id oriented/directed
        :param acyclic: if the graph id oriented/directed and acyclic
        :param loops: if the loops are permitted
        :param multi: if multiple edges are permitted between 2 nodes
        :param gprops: graph properties
        """
        self._directed = directed or acyclic
        self._loops = loops and not acyclic
        self._acyclic = acyclic
        self._props = {} | gprops

        self._nodes: dict[NODE_TYPE, dict[str, Any]] = {}
        self._edges: IEdges = None

        if acyclic:
            self._edges: IEdges = DAGEdges(multi)
        elif directed:
            self._edges: IEdges = DEdges(self._loops, multi)
        else:
            self._edges: IEdges = UEdges(self._loops, multi)

        if "name" not in self._props:
            self._props["name"] = "G"

        # self._succ = self._edges.succ
        # self._pred = self._edges.pred

        # local cache used to save results of some complex computation
        # Cleared on EACH change
        self.__netx_cache__: dict[str, dict] = defaultdict(lambda : dict())
    # end

    def copy(self):
        # networks
        C = Graph(directed=self.is_directed(),
                  loops=self.has_loops(),
                  multi=self.is_multigraph(),
                  acyclic=self.is_acyclic(),
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
                directed=self.is_directed(),
                acyclic=self.is_acyclic(),
                loops=self.has_loops(),
                multi = self.is_multigraph(),
            )
        else:
            return self._props

    def is_directed(self) -> bool:
        # networkx
        return self._directed

    def is_multigraph(self) -> bool:
        # networkx
        return self._edges.multi

    def has_loops(self) -> bool:
        return self._edges.loops

    def is_acyclic(self) -> bool:
        return self._acyclic

    def is_dag(self) -> bool:
        return self._directed and self._acyclic

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
        # return max(self._nodes.keys()) + 1

    def number_of_nodes(self) -> int:
        # networkx
        # return len(self._nodes)
        return self.order()

    # -----------------------------------------------------------------------

    @cached_property
    def nodes(self):
        # networkx
        # return self._nodes.keys()
        return NodesView(self._nodes)

    def add_node(self, n: NODE_TYPE, **nprops):
        # networkx
        return self._add_node(n, nprops)

    def add_nodes_from(self, nlist: Union[Collection[NODE_TYPE], Iterator[NODE_TYPE]], **nprops):
        # ([n1, ...], prop1=value1, ...)
        # ([(n1, {prop1: value1}), ...],
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
        if n in self._nodes:
            del self._nodes[n]
        self._edges.remove_node(n)

    def remove_nodes_from(self, nlist: Collection[NODE_TYPE]):
        # networkx
        for n in nlist:
            self.remove_node(n)

    def has_node(self, n: NODE_TYPE) -> bool:
        # networkx
        return n in self._nodes

    def neighbors(self, n: NODE_TYPE, inbound: Optional[bool] = False) -> set[NODE_TYPE]:
        """
        Neighbours of the specified node.
        If the graph is not directed, the parameter 'inbound' has no effect.
        For directed graphs, 'inbound' can be:

             None: in and out edges
            False: out edges
             True: in edges

        :param n:
        :param inbound: None (in/out), False (out), True (in)
        :return:  set of nodes
        """
        # networkx
        return set(self._edges.neighbors(n, inbound))

    def successors(self, n: NODE_TYPE) -> set[NODE_TYPE]:
        return self.neighbors(n, inbound=False)

    def predecessors(self, n: NODE_TYPE) -> set[NODE_TYPE]:
        return self.neighbors(n, inbound=True)

    def nbunch_iter(self, source: Optional[Union[list,tuple,NODE_TYPE]]) -> Iterator[NODE_TYPE]:
        # networkx
        if source is None:
            return self.nodes()
        if isinstance(source, Union[list, tuple]):
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
        elif len(nprops) > 0:
            self._nodes[n] |= nprops
        self.__netx_cache__.clear()
        return self

    # -----------------------------------------------------------------------
    # Operations/edges
    #   G.edges
    #   e in G.edges

    @cached_property
    def edges(self):
        # networkx
        # return self._edges.keys()
        return EdgesView(self._edges, EdgesView.EDGES)

    @cached_property
    def in_edges(self):
        # networkx (DiGraph)
        return EdgesView(self._edges, EdgesView.IN_EDGES)

    @cached_property
    def out_edges(self):
        # networkx (DiGraph)
        return EdgesView(self._edges, EdgesView.OUT_EDGES)

    def size(self) -> int:
        # networkx
        return len(self._edges)

    def number_of_edges(self) -> int:
        # networkx
        return len(self._edges)

    # -----------------------------------------------------------------------

    def add_edge(self, u: NODE_TYPE, v: NODE_TYPE, **eprops):
        # networkx
        self._add_edge(u, v, eprops)
        return self

    def add_edges_from(self, elist: Union[list[EDGE_TYPE], Iterator[EDGE_TYPE]], **eprops):
        # networkx
        for e in elist:
            if len(e) == 2:
                u, v = e
                self._add_edge(u, v, eprops)
                continue

            if isinstance(e[-1], dict):
                uvprops = e[-1]
                e = e[:-1]
            else:
                uvprops = {}

            n = len(e)
            for i in range(n-1):
                u = e[i]
                v = e[i+1]
                self._add_edge(u, v, eprops | uvprops)
        return self

    # def add_weighted_edges_from(self, ebunch_to_add, weight="weight", **attr):

    def remove_edge(self, u: NODE_TYPE, v: NODE_TYPE):
        # networkx
        self._edges.remove_edge(u, v)
        return self

    def remove_edges_from(self, elist: list[EDGE_TYPE]):
        # networkx
        for e in elist:
            self.remove_edge(*e)
        return self

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
        self.__netx_cache__.clear()
        return self

    # -----------------------------------------------------------------------
    # For undirected graphs, succ and pred are the same

    @cached_property
    def adj(self) -> AdjacencyView: # dict[NODE_TYPE, set[NODE_TYPE]]:
        # networkx
        return AdjacencyView(self._edges, self._edges.adj)

    @cached_property
    def succ(self) -> AdjacencyView: # dict[NODE_TYPE, set[NODE_TYPE]]:
        # networkx (DiGraph)
        return AdjacencyView(self._edges, self._edges.succ)

    @cached_property
    def pred(self) -> AdjacencyView: # dict[NODE_TYPE, set[NODE_TYPE]]:
        # networkx (DiGraph)
        return AdjacencyView(self._edges, self._edges.pred, True)

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

    @cached_property
    def degree(self, multi=False):
        # networkx
        return DegreeView(self._edges, DegreeView.DEGREE)

    @cached_property
    def in_degree(self, multi=False):
        # networkx (DiGraph)
        return DegreeView(self._edges, DegreeView.IN_DEGREE)

    @cached_property
    def out_degree(self, multi=False):
        # networkx (DiGraph)
        return DegreeView(self._edges, DegreeView.OUT_DEGREE)

    # -----------------------------------------------------------------------

    def clone(self, name=None):
        # networkx
        G = Graph(
            name=name,
            directed=self.is_directed(),
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
        super().__init__(directed=True, loops=loops, **gprops)


class DirectAcyclicGraph(Graph):
    def __init__(self, **gprops):
        super().__init__(acyclic=True, directed=True, loops=False, **gprops)


class MultiGraph(Graph):
    def __init__(self, loops=False, **gprops):
        super().__init__(multi=True, loops=loops, **gprops)


class MultiDiGraph(Graph):
    def __init__(self, loops=False, **gprops):
        super().__init__(directed=True, multi=True, loops=loops, **gprops)



NetxGraph = Graph
NetxDiGraph = DiGraph


# ---------------------------------------------------------------------------
# Graph classes
# ---------------------------------------------------------------------------

def is_netx_graph(G):
    gclass = G.__class__
    return gclass in {Graph}


def is_networkx_graph(G):
    gclass = G.__class__
    return gclass in {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}


# ---------------------------------------------------------------------------
# create_like
# ---------------------------------------------------------------------------

def create_like(G):
    if isinstance(G, Graph):
        return Graph(directed=G.is_directed(), loops=G.has_loops(), multi=G.is_multigraph(), acyclic=G.is_acyclic())
    if isinstance(G, nx.DiGraph):
        return nx.DiGraph()
    if isinstance(G, nx.MultiDiGraph):
        return nx.MultiDiGraph()
    if isinstance(G, nx.MultiGraph):
        return nx.MultiGraph()
    if isinstance(G, nx.Graph):
        return nx.Graph()
    else:
        raise TypeError(f"Unsupported graph type: {type(G)}")


# ---------------------------------------------------------------------------
# print_graph_stats
# ---------------------------------------------------------------------------

def print_graph_stats(G: nx.Graph):
    n = G.order()
    m = G.size()
    if G.is_directed():
        print(f"G={{|V|={n}, |V|={m}, direct}}")
    else:
        print(f"G={{|V|={n}, |V|={m}}}")
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

