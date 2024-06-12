from typing import List, Dict, Set, Tuple, Sized, Union
from collections import defaultdict

import networkx as nx

__version__ = "1.0.0"

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


class IEdges(Sized):

    @property
    def adj(self) -> Dict[int, List[int]]: return {}

    @property
    def succ(self) -> Dict[int, List[int]]: return {}

    @property
    def prec(self) -> Dict[int, List[int]]: return {}

    def __getitem__(self, item: Tuple[int, int]) -> List[int]: return []
    def __setitem__(self, item: Tuple[int, int], value: List[int]): ...
    def __len__(self): return 0


class uedges(dict, IEdges):
    """Undirected edges dictionary"""

    def __init__(self, loops=False):
        super().__init__()
        self.loops = loops
        self._adj: Dict[int, List[int]] = defaultdict(lambda: list())

    @property
    def adj(self):
        return self._adj

    @property
    def succ(self):
        return self._adj

    @property
    def prec(self):
        return self._adj

    def __getitem__(self, item):
        u, v = item
        if u > v:
            u, v = v, u
        return super().__getitem__((u,v))

    def __setitem__(self, item, value):
        u, v = item
        if u > v:
            u, v = v, u
        if not self.loops and u == v:
            return None
        elif super().__contains__((u,v)):
            return super().__setitem__((u, v), value)
        else:
            if u not in self._adj or v not in self._adj[u]:
                self._adj[u].append(v)
                self._adj[v].append(u)
            # end
            return super().__setitem__((u,v), value)
    # end
# end


class dedges(dict, IEdges):
    """Directed edges dictionary"""

    def __init__(self, loops=False):
        super().__init__()
        self.loops = loops
        self._succ: Dict[int, List[int]] = defaultdict(lambda: list())
        self._prec: Dict[int, List[int]] = defaultdict(lambda: list())

    @property
    def adj(self) -> Dict[int, List[int]]:
        return self._succ

    @property
    def succ(self) -> Dict[int, List[int]]:
        return self._succ

    @property
    def prec(self) -> Dict[int, List[int]]:
        return self._prec

    def __getitem__(self, item):
        u, v = item
        return super().__getitem__((u,v))

    def __setitem__(self, item, value):
        u, v = item
        if not self.loops and u == v:
            return None
        elif super().__contains__((u,v)):
            return super().__setitem__((u, v), value)
        else:
            self._succ[u].append(v)
            self._prec[v].append(u)
            return super().__setitem__((u,v), value)
    # end
# end


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

    def __init__(self, direct=False, loops=False, multi=False, **gprops):
        """

        :param direct: if the graph id oriented
        :param loops: if the loops are supported
        :param multi: if there are multiple edges between 2 nodes
        :param gprops: graph properties
        """
        self._direct = direct
        self._loops = loops
        self._multi = multi
        self._props = gprops

        self._nodes: Dict[int, dict] = {}
        if direct:
            self._edges: IEdges = dedges(loops)
        else:
            self._edges: IEdges = uedges(loops)

        if "name" not in self._props:
            self._props["name"] = "G"
    # end

    # -----------------------------------------------------------------------
    # Properties

    @property
    def name(self) -> str:
        return self._props["name"]

    @property
    def properties(self) -> Dict:
        return self._props | dict(
            direct=self._direct,
            multi=self._multi,
            loops=self._loops
        )

    @property
    def is_directed(self) -> bool:
        return self._direct

    @property
    def nodes(self) -> Dict[int, Dict]:
        return self._nodes

    @property
    def edges(self) -> IEdges:
        return self._edges

    @property
    def adj(self) -> Dict[int, List[int]]:
        return self._edges.adj

    @property
    def succ(self) -> Dict[int, List[int]]:
        return self._edges.succ

    @property
    def prec(self) -> Dict[int, List[int]]:
        return self._edges.prec

    # -----------------------------------------------------------------------
    # Operations

    def add_node(self, n, **nprops):
        if n not in self._nodes: 
            self._nodes[n] = nprops
            self.adj[n] = []
        else:
            pass
        return self

    def add_edge(self, u, v, **eprops):
        if not self._direct and u > v:
            u, v = v, u

        if not self._check_edge(u, v):
            return

        self.add_node(u)
        self.add_node(v)

        if v not in self._edges.adj[u]:
            self._edges[(u, v)] = [eprops] if self._multi else eprops
        elif self._multi:
            self._edges[(u, v)].append(eprops)
        # end
        return self

    def _check_edge(self, u, v):
        # check if u == v (loop)
        # check if (u,v) is an edge already present
        #   (multiple edges)
        # check for dag
        if not self._loops and u == v:
            return False
        if not self._multi and (u, v) in self._edges:
            return False
        return True

    # -----------------------------------------------------------------------
    # Node degree

    def degree(self, n, multi=False) -> int:
        if self._direct:
            return self.in_degree(n, multi=multi) + self.out_degree(n, multi=multi)
        else:
            return self.out_degree(n, multi=multi)

    def out_degree(self, u, multi=False) -> int:
        if not self._multi or not multi:
            return len(self._edges.succ[u])
        else:
            deg = 0
            for v in self._edges.succ[u]:
                deg += len(self._edges[(u, v)])
            return deg

    def in_degree(self, v, multi=False) -> int:
        if not self._multi or not multi:
            return len(self._edges.prec[v])
        else:
            deg = 0
            for u in self._edges.prec[v]:
                deg += len(self._edges[(u, v)])
            return deg

    # -----------------------------------------------------------------------
    # Node degree

    def __getitem__(self, n_e: Union[int, Tuple[int, int]]) -> List[int]:
        """
        :param n_e: node or edge
        :return:
        """
        if not isinstance(n_e, tuple):
            if self._direct:
                return self._edges.succ[n_e]
            else:
                return self._edges.adj[n_e]
        else:
            return self._edges[n_e]

    def __iter__(self):
        return iter(self._edges.adj)

    def __repr__(self):
        nv = len(self._nodes)
        ne = len(self._edges)
        return f"{self.name}=(|V|={nv}, |E|={ne})"
# end


class DiGraph(Graph):
    # def __init__(self, direct=False, loops=False, multi=False, **gprops)
    def __init__(self, loops=False, **gprops):
        super().__init__(direct=True, loops=loops, **gprops)
# end


class MultiGraph(Graph):
    # def __init__(self, direct=False, loops=False, multi=False, **gprops)
    def __init__(self, loops=False, **gprops):
        super().__init__(multi=True, oops=loops, **gprops)
# end


class MultiDiGraph(Graph):
    # def __init__(self, direct=False, loops=False, multi=False, **gprops)
    def __init__(self, loops=False, **gprops):
        super().__init__(direct=True, multi=True, loops=loops, **gprops)
# end

