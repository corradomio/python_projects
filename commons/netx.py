from typing import List, Dict, Set, Tuple
from collections import defaultdict

import networkx as nx

__version__ = "1.0.0"


class IEdges:

    @property
    def adj(self) -> Dict[int, List[int]]: pass

    @property
    def succ(self) -> Dict[int, List[int]]: pass

    @property
    def prec(self) -> Dict[int, List[int]]: pass

    def __getitem__(self, item: Tuple[int, int]) -> List[int]: pass
    def __setitem__(self, item: Tuple[int, int], value: List[int]): pass


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

    def __init__(self, direct=False, loops=False, **kwargs):
        self._direct = direct
        self._loops = loops
        self._graph = kwargs
        self._nodes: Dict[int, dict] = dict()
        if direct:
            self._edges: IEdges = dedges(loops)
        else:
            self._edges: IEdges = uedges(loops)
        # end
        if "name" not in self._graph:
            self._graph["name"] = "G"
    # end

    @property
    def name(self) -> str:
        return self._graph["name"]

    @property
    def graph(self):
        return self._graph

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    @property
    def adj(self):
        return self._edges.adj

    @property
    def succ(self):
        return self._edges.succ

    @property
    def prec(self):
        return self._edges.prec

    def is_directed(self) -> bool:
        return self._direct

    def add_node(self, n, **kwargs):
        if n not in self._nodes: 
            self._nodes[n] = kwargs
            self.adj[n] = []
    # end

    def add_edge(self, u, v, **kwargs):
        if not self._direct and u > v:
            u, v = v, u
        self.check_edge(u, v)

        self.add_node(u)
        self.add_node(v)

        if v in self._edges.adj[u]:
            return

        self._edges[(u, v)] = kwargs
        # end
    # end

    def __getitem__(self, n_e):
        if isinstance(n_e, int):
            if self._direct:
                return self._edges.succ[n_e]
            else:
                return self._edges.adj[n_e]
        else:
            return self._edges[tuple(n_e)]

    def __iter__(self):
        return iter(self._edges.adj)

    def check_edge(self, u, v):
        # check if u == v (loop)
        # check if (u,v) is an edge already present
        #   (multiple edges)
        pass

    def __repr__(self):
        nv = len(self._nodes)
        ne = len(self._edges)
        return f"{self.name}=(|V|={nv}, |E|={ne})"
# end


def coarsening_graph(g: Graph, partitions: List[List[int]], create_using=None, direct=False, **kwargs) -> Graph:
    """
    Create a coarse graph using the partitions as 'super' nodes and edges between
    partitions i and j if there exist a node in partition i connected to a node in partition j

    :param g: original graph
    :param partitions: partitions
    :param create_using: which graph to populate, or None
    :param direct: if to create a direct graph
    :param kwargs: extra parameters passed to the created graph
    :return: a coarsed graph
    """
    assert isinstance(partitions, list)

    # in_partition[u] -> partition of u
    def in_partition(nv):
        in_part = [0]*nv
        c = 0
        for partition in partitions:
            for u in partition:
                in_part[u] = c
            c += 1
        return in_part

    # create an empty graph
    if create_using is not None:
        coarsed = create_using
    elif direct:
        coarsed = nx.DiGraph(**kwargs)
    else:
        coarsed = nx.Graph(**kwargs)
    name = g.graph["name"] if "name" in g.graph else "G"
    # if "name" not in coarsed.graph:
    coarsed.graph["name"] = f"coarsed-{name}"

    # force partitions to be a list
    partitions = list(partitions)
    in_part = in_partition(len(g.nodes))
    for u, v in list(g.edges):
        cu = in_part[u]
        cv = in_part[v]
        if cu != cv:
            coarsed.add_edge(cu, cv)
    # end
    return coarsed
# end


def closure_coarsening_graph(g: nx.DiGraph, create_using=None, direct=True, **kwargs) -> nx.Graph:
    """
    Create a coarse graph using the following protocol

        1) for each node creates the transitive closure
        2) create an edge from closure i and closure j if closure i is a proper subset of closure j
        3) apply a transitive reduction

    :param g: original graph
    :param create_using: which graph to populate, or None
    :param direct: if to create a direct graph
    :param kwargs: extra parameters passed to the created graph
    :return: a coarsed graph
    """
    assert g.is_directed()

    def closure_of(u: int, closures: Dict[int, Set[int]]) -> Set[int]:
        visited: Set[int] = set()
        tovisit: List[int] = [u]
        while len(tovisit) > 0:
            u: int = tovisit.pop()
            if u in visited:
                continue

            if u in closures:
                visited.update(closures[u])
            else:
                visited.add(u)
                tovisit += list(g.succ[u])
        # end
        return visited
    # end

    def is_subset_of(s1: Set[int], s2: Set[int]) -> bool:
        if len(s1) >= len(s2):
            return False
        for u in s1:
            if u not in s2:
                return False
        return True
    # end

    # create an empty graph
    if create_using is not None:
        coarsed = create_using
    elif direct:
        coarsed = nx.DiGraph(**kwargs)
    else:
        coarsed = nx.Graph(**kwargs)
    name = g.graph["name"] if "name" in g.graph else "G"
    coarsed.graph["name"] = f"closure-coarsed-{name}"

    # compute the closures
    closures: Dict[int, Set[int]] = dict()
    for u in g:
        closures[u] = closure_of(u, closures)
    # end

    # compute the simplified graph
    for u in closures:
        cu = closures[u]
        for v in closures:
            cv = closures[v]

            if len(cu) == len(cv):
                continue
            if is_subset_of(cv, cu):
                coarsed.add_edge(u, v)
        # end
    # end

    # apply the transitive reduction
    reducted = nx.transitive_reduction(coarsed)


    reducted.graph.update(coarsed.graph)

    return coarsed
# end


def read_vecsv(path: str, comments="#", header=True, separator=",", create_using=None, direct=True) -> nx.Graph:
    # -vertices.csv
    # -edges.csv
    def vecsv_strip(path: str) -> str:
        suffix = "-vertices.csv"
        if path.endswith(suffix):
            return path[0:-len(suffix)]
        suffix = "-edges.csv"
        if path.endswith(suffix):
            return path[0:-len(suffix)]
        suffix = ".csv"
        if path.endswith(suffix):
            return path[0:-len(suffix)]
        else:
            return path
    # end

    def vecsv_vertices_file(path: str) -> str:
        return vecsv_strip(path) + "-vertices.csv"

    def vecsv_edges_file(path: str) -> str:
        return vecsv_strip(path) + "-edges.csv"

    def parse(s):
        try:
            return int(s)
        except:
            pass
        try:
            return float(s)
        except:
            pass
        return s
    # end

    vfile = vecsv_vertices_file(path)
    efile = vecsv_edges_file(path)

    if create_using is not None:
        g = create_using
    elif direct:
        g = nx.DiGraph()
    else:
        g = nx.Graph()

    # read vertices
    columns = None
    with open(vfile) as vfin:
        for line in vfin:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(comments):
                continue
            if columns is None:
                if header:
                    columns = line.split(separator)
                    continue
                else:
                    ncols = len(line.split(separator))
                    columns = [f"c{i+1:02}" for i in range(ncols)]
            # end
            props = list(map(parse, line.split(separator)))
            node = props[0]

            g.add_node(node)
            nattrs = dict()
            for i in range(1, len(columns)):
                nattrs[columns[i]] = props[i]
            g.nodes[node].update(nattrs)
        # end

    # read edges
    columns = None
    with open(efile) as efin:
        for line in efin:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith(comments):
                continue
            if columns is None:
                if header:
                    columns = line.split(separator)
                    continue
                else:
                    ncols = len(line.split(separator))
                    columns = [f"c{i + 1:02}" for i in range(ncols)]
            # end
            edge = list(map(parse, line.split(separator)))
            source = edge[0]
            target = edge[1]
            g.add_edge(source, target)
        # end
    return g
# end


def print_graph_stats(g:nx.Graph):
    n = len(g.nodes)
    m = len(g.edges)
    if g.is_directed():
        print(f"G={{V: {n}, E: {m}}}")
    else:
        print(f"G={{V: {n}, E: {m}}}, directed")
# end
