from collections import defaultdict, deque
from typing import Optional

edge = tuple[int, int]


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
    def adj(self) -> dict[int, list[int]]:
        """
        Adjacency list:
            u -> [v1, v2, ...]
        """
        return {}

    @property
    def succ(self) -> dict[int, list[int]]:
        """
        Successors of a node
        """
        return {}

    @property
    def pred(self) -> dict[int, list[int]]:
        """
        Predecessors of a node
        :return:
        """
        return {}

    def neighbors(self, n: int, inbound: Optional[bool]) -> list[int]:
        """
        Neighbors of a node
        :param n: node
        :param inbound: if to consider the inbound edges only
        :return:
        """
        return []

    def add_edge(self, u, v, eprops):
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

    def _check_edge(self, u, v):
        # check if u == v (loop)
        # check if (u,v) is an edge already present
        #   (multiple edges)
        # check for dag
        if not self.loops and u == v:
            return False
        if not self.multi and (u, v) in self:
            return False
        return True

    def out_degree(self, u, multi=False) -> int:
        if not self.multi or not multi:
            return len(self.succ[u])
        else:
            deg = 0
            for v in self.succ[u]:
                deg += len(self[(u, v)])
            return deg

    def in_degree(self, v, multi=False) -> int:
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


class UEdges(IEdges):
    """Undirected edges dictionary"""

    def __init__(self, loops: bool, multi: bool):
        super().__init__(loops, multi)
        self._adj: dict[int, list[int]] = defaultdict(lambda: list())

    @property
    def adj(self) -> dict[int, list[int]]:
        return self._adj

    @property
    def succ(self) -> dict[int, list[int]]:
        return self._adj

    @property
    def pred(self) -> dict[int, list[int]]:
        return self._adj

    def add_edge(self, u, v, eprops):
        if u > v:
            u, v = v, u
        super().add_edge(u, v, eprops)

    def neighbors(self, n: int, inbound: Optional[bool]) -> list[int]:
        return self._adj[n]

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

        self._succ: dict[int, list[int]] = defaultdict(lambda: list())
        self._prec: dict[int, list[int]] = defaultdict(lambda: list())

    @property
    def adj(self) -> dict[int, list[int]]:
        return self._succ

    @property
    def succ(self) -> dict[int, list[int]]:
        return self._succ

    @property
    def pred(self) -> dict[int, list[int]]:
        return self._prec

    def add_edge(self, u, v, eprops):
        super().add_edge(u, v, eprops)

    def neighbors(self, n: int, inbound: Optional[bool]) -> list[int]:
        if inbound is None:
            return self._prec[n] + self._succ[n]
        elif inbound:
            return self._prec[n]
        else:
            return self._succ[n]

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


class DagEdges(DEdges):
    def __init__(self, multi: bool):
        super().__init__(False, multi)

    def add_edge(self, u, v, eprops):
        if self.has_path(v, u):
            return
        else:
            super().add_edge(u, v, eprops)

    def has_path(self, u: int, v: int) -> bool:
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

