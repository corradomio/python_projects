#
# Approximations of Shapley Value/Interaction Index, Banzhaf Value/Interaction Index
#
from time import time
from typing import Tuple, Union
from numpy import ndarray, zeros
from randomx import RandomPermutations, RandomSets, WeightedRandomSets
from iset import iset, imembers, iremove, iadd, idiff, icard
from stdlib.imathx import ilog2


# ---------------------------------------------------------------------------
# Approximation Information
# ---------------------------------------------------------------------------

class ApproxInfo:

    def __init__(self, n_jobs: Union[None, int, Tuple[int, int]]=None, random_state=None):
        self.n_jobs = n_jobs
        self.random_state = random_state if random_state is not None else int(time())
    # end

    def update(self):
        self.random_state += 1000
# end


# ---------------------------------------------------------------------------
# Shapley Approximation
# ---------------------------------------------------------------------------

class ShapleyApproxInfo(ApproxInfo):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.generated = set()
        self.lv = zeros((n, n), dtype=float)
        self.sv = zeros(n, dtype=float)
        self.c = zeros(n, dtype=int)
    # end
# end


def _shapley_approx_perm(xi: ndarray, P: tuple) -> (ndarray, ndarray):
    n = len(P)
    sv = zeros(n, dtype=float)
    c = zeros(n, dtype=int)

    for k in range(n):
        i = P[k]
        Si = iset(P[0:k + 1])
        S = iremove(Si, i)
        sv[i] += xi[Si] - xi[S]
        c[i] += 1
    return sv, c
# end


def shapley_value_approx_perms(xi: ndarray, m: int, ainfo: ShapleyApproxInfo) -> ndarray:
    """
    :param xi: set function
    :param m: n of sets
    """
    p = len(xi)
    n = ilog2(p)

    # results = Parallel(n_jobs=ainfo.n_jobs)(delayed(_shapley_approx_perm)(xi, P) for P in RandomPermutations(n, m))
    results = [_shapley_approx_perm(xi, P) for P in RandomPermutations(n, m)]
    ainfo.update()

    sv = ainfo.sv
    c = ainfo.c

    for svi, ci in results:
        sv += svi
        c += ci

    nc = c.copy()
    nc[nc == 0] = 1
    return sv/nc
# end


def _shapley_approx_set(xi, S) -> (ndarray, ndarray):
    p = len(xi)
    n = ilog2(p)
    N = p - 1

    lv = zeros((n, n), dtype=float)
    c = zeros(n, dtype=int)

    S = iset(S)
    s = icard(S)
    R = idiff(N, S)
    for i in imembers(R):
        Si = iadd(S, i)
        lv[s, i] += xi[Si] - xi[S]
        c[s] += 1
    return lv, c
# end


def shapley_value_approx_partial(xi: ndarray, m: int, ainfo: ApproxInfo) -> (ndarray, ndarray):
    p = len(xi)
    n = ilog2(p)

    lv = ainfo.lv
    c = ainfo.c

    # results = Parallel(n_jobs=ainfo.n_jobs)(delayed(_shapley_approx_set)(xi, S)
    #                                         for S in WeightedRandomSets(n, m, plevels=[1]*n))
    results = [_shapley_approx_set(xi, S) for S in WeightedRandomSets(n, m, plevels=[1]*n)]
    ainfo.update()
    for tlv, tc in results:
        lv += tlv
        c += tc

    sv = zeros(n, dtype=float)
    nc = c.copy()
    nc[nc == 0] = 1
    for i in range(n):
        sv[i] = (lv[:, i] / nc).sum()
    return sv
# end


# ---------------------------------------------------------------------------
# Banzhaf Approximation
# ---------------------------------------------------------------------------

class BanzhafApproxInfo(ApproxInfo):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.generated = set()
        self.n = n
        self.bv = zeros(n, dtype=float)
        self.c = zeros(n, dtype=int)
    # end
# end


def _banzhaf_approx_set(xi: ndarray, S: set) -> (ndarray, ndarray):
    p = len(xi)
    N = p - 1
    n = ilog2(p)

    pv = zeros(n, dtype=float)
    c = zeros(n, dtype=int)

    S = iset(S)
    R = idiff(N, S)
    for i in imembers(R):
        Si = iadd(S, i)
        pv[i] += xi[Si] - xi[S]
        c[i] += 1
    # end

    return pv, c
# end


def banzhaf_value_approx_partial(xi: ndarray, m: int, ainfo: BanzhafApproxInfo) -> (ndarray, ndarray):
    n = ainfo.n
    # results = Parallel(n_jobs=ainfo.n_jobs)(delayed(_banzhaf_approx_set)(xi, S)
    #                                         for S in RandomSets(n, m, generated=ainfo.generated))
    results = [_banzhaf_approx_set(xi, S) for S in RandomSets(n, m, generated=ainfo.generated)]
    ainfo.update()

    bv = ainfo.bv
    c = ainfo.c
    for pvi, ci in results:
        bv += pvi
        c += ci

    nc = c.copy()
    nc[nc == 0] = 1
    return bv / nc
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
