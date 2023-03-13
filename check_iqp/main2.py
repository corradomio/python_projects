from typing import List
from path import Path as path
import numpy as np
from mathx import ilog2
from iset import iset
from numpy import ndarray
from pprint import pprint
from random import random, randint, Random
from optimizers import *


np.set_printoptions(suppress=True, precision=4, linewidth=1024)


DATA_DIR = "D:\\Projects_PhD\\sfun_evaluate\\iidxs_mat"
FILE_MAT = "autism_dt_sv"


def load_mat(fname, n=0):
    m = np.loadtxt(fname, delimiter=",", dtype=float)
    if n > 0:
        m = m[0:n, 0:n]
    return m
# end


# ---------------------------------------------------------------------------
# View splitting
# ---------------------------------------------------------------------------

class ViewSolution(IOptimalSolution):

    def __init__(self, n, v, seed=None):
        self.r = Random(seed)
        self.n = n
        self.v = v
        self.view = np.zeros(n, dtype=int)
        self.xpi = np.zeros(n * v, dtype=int)
        self.random()
    # end

    def views(self) -> List[tuple]:
        """return the features contained in each view"""
        n = self.n
        v = self.v
        vp = [[] for i in range(v)]
        for i in range(n):
            p = self.view[i]
            vp[p].append(i)
        # end
        return sorted(map(lambda v: tuple(v), vp))
    # end

    def copy(self) -> "ViewSolution":
        vs = ViewSolution(self.n, self.v, self.r.random())
        vs.view[:] = self.view
        vs.xpi[:] = self.xpi
        return vs
    # end

    def random(self) -> "ViewSolution":
        """generate a random solution"""
        r = self.r
        n = self.n
        v = self.v
        for i in range(n):
            p = r.randint(0, v - 1)
            self.view[i] = p
        self.xpi[:] = 0
        for i in range(n):
            p = self.view[i]
            self.xpi[p * n + i] = 1
        return self
    # end

    def tweak(self, n_tweaks=1) -> "ViewSolution":
        """apply nc changes"""
        n = self.n
        v = self.v
        xpi = self.xpi
        r = self.r
        for k in range(n_tweaks):
            i = r.randint(0, n - 1)
            op = xpi[i]  # old view
            np = r.randint(0, v - 1)  # new view
            self.view[i] = np
            self.xpi[op * n + i] = 0
            self.xpi[np * n + i] = 1
        # end
        return self
    # end

    # -----------------------------------------------------------------------

    def quality_on_pii(self, m: ndarray) -> float:
        n = self.n
        q = 0.

        for i in range(n):
            q += m[i, i]
            for j in range(i+1, n):
                pi = self.view[i]
                pj = self.view[j]
                if pi == pj:
                    q += m[i, j]
                else:
                    q -= m[i, j]
            # end
        # end
        return q
    # end

    def is_ideal_on_pii(self, m: ndarray, n_tweaks: int) -> bool:
        r = self.r
        n =self.n
        qS = self.quality_on_pii(m)
        for i in range(n_tweaks):
            nc = r.randint(0, n)
            T = self.copy().tweak(nc)
            qT = T.quality_on_pii(m)
            if qT > qS:
                return False
        return True
    # end

    # -----------------------------------------------------------------------

    def quality_on_sf(self, xi: ndarray):
        views = self.views()
        q = 0.
        for v in views:
            q += xi[iset(v)]
        return q
    # end

    def is_ideal_on_sf(self, xi: ndarray, nv: int) -> bool:
        r = self.r
        n = self.n
        qS = self.quality_on_sf(xi)
        for i in range(nv):
            nc = r.randint(0, n)
            T = self.copy().tweak(nc)
            qT = T.quality_on_sf(xi)
            if qT > qS:
                return False
        return True
    # end

    # -----------------------------------------------------------------------

    def eval(self, xi):
        views = self.views()
        q = 0.
        for v in views:
            q += xi[iset(v)]
        return round(q, 6)

    def to_views_quality(self, xi) -> List[float]:
        views = self.views()
        ql = []
        for v in views:
            ql.append(xi[iset(v)])
        return list(map(lambda x: round(x, 6), ql))
    # end

    def to_views_solution(self):
        vl = []
        views = self.views()
        for v in views:
            vl.append(":".join(map(str, v)))
        return vl
    # end

    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return str(self.view)

    def __eq__(self, other: "ViewSolution") -> bool:
        return np.all(self.view == other.view)
# end


class ViewSplitting(ISingleStateMethod):

    def __init__(self, n: int, v: int, n_changes: int=1, n_validate: int=0, **kwargs):
        """
            max(x)  1/2 x^ Q x + c^T x

        :param v: n of views
        :param Q:
        :param c:
        :param n_changes: n of changes for each tweak
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.n = n
        self.v = v
        self.n_changes = n_changes
        self.n_validate = n_validate if n_validate > 0 else n+v

    def initial_solution(self) -> ViewSolution:
        """Generate a single initial solution"""
        return ViewSolution(self.n, self.v)

    def copy(self, S: ViewSolution) -> ViewSolution:
        """Create a copy of the solution"""
        return S.copy()

    def tweak(self, S: ViewSolution):
        """Modify the current solution with a small number of changes"""
        return S.tweak(self.n_changes)

    # def is_ideal_solution(self, S: ViewSolution, qS: float) -> bool:
    #     """Check if the solution is ideal"""
    #     return S.is_ideal(self.m, self.n_validate)

    # def quality(self, S: ViewSolution):
    #     """Evaluate the quality of the solution"""
    #     return S.quality(self.m)
# end


# ---------------------------------------------------------------------------
# View splitting based on Power&Interaction indices
# ---------------------------------------------------------------------------

class ViewSplittingOnPII(ViewSplitting):

    def __init__(self, m: ndarray, *args, **kwargs):
        super().__init__(len(m), *args, **kwargs)
        assert isinstance(m, ndarray) and len(m.shape) == 2
        self.m = m

    def is_ideal_solution(self, S: ViewSolution, qS: float) -> bool:
        """Check if the solution is ideal"""
        return S.is_ideal_on_pii(self.m, self.n_validate)

    def quality(self, S: ViewSolution):
        """Evaluate the quality of the solution"""
        return S.quality_on_pii(self.m)
# end


# ---------------------------------------------------------------------------
# View splitting based on SetFunction
# ---------------------------------------------------------------------------

class ViewSplittingOnSF(ViewSplitting):

    def __init__(self, xi: ndarray, *args, **kwargs):
        super().__init__(ilog2(len(xi)), *args, **kwargs)
        assert isinstance(xi, ndarray) and len(xi.shape) == 1
        self.xi = xi

    def is_ideal_solution(self, S: ViewSolution, qS: float) -> bool:
        """Check if the solution is ideal"""
        return S.is_ideal_on_sf(self.xi, self.n_validate)

    def quality(self, S: ViewSolution):
        """Evaluate the quality of the solution"""
        return S.quality_on_sf(self.xi)
# end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    fname = path(DATA_DIR).joinpath(FILE_MAT+".csv")

    m = load_mat(fname)
    n = len(m)
    v = 2

    for i in range(3):
        print("--", i+1, "--")
        opt = TabuSearch(delegate=ViewSplittingOnPII(m, v), seed=random(), n_loop=n*10)
        sol = opt.find_optimum(n_epochs=n*500)
        pprint(sol)
        pprint(sol[0].views())
    print("-- end --")
# end


if __name__ == "__main__":
    main()
