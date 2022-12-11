#
# Set Functions: Feature Selection
#
#   .
from numpy import ndarray
from mathx import INF
from iset import *
from .sfun_fun import SetFunction


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def tosets(P: tuple) -> List[tuple]:
    n = len(P)
    L = []
    for i in range(n+1):
        S = P[0:i]
        L.append(tuple(sorted(S)))
    return L
# end


def argmaxadd(xi, S: list, Ul: list):
    v = -INF
    s = -1
    U = iset(Ul)
    for i in S:
        T = iadd(U, i)
        if xi[T] > v:
            v = xi[T]
            s = i
    # end
    return s
# end


def argmaxrem(xi, Sl: list):
    v = -INF
    s = -1
    S = iset(Sl)
    for i in imembers(S):
        T = iremove(S, i)
        if xi[T] > v:
            v = xi[T]
            s = i
    # end
    return s
# end


# ---------------------------------------------------------------------------
# Greedy Methods
# ---------------------------------------------------------------------------

def sequential_forward_selection(xi: ndarray) -> tuple:
    """
    :param xi: set function
    :param s: n of features to select
    :return: set of selected features
    """
    p = len(xi)
    N = p - 1
    selected = []    # iset([])
    available = N   # iset([0,1,...,n-1])

    while icount(available) > 0:
        f = -INF
        s = ilowbit(available)
        for i in imembers(available):
            S = iset(selected + [i])
            if xi[S] > f:
                s = i
                f = xi[S]
            # end
        # end
        selected = selected + [s]
        available = iremove(available, s)
    # end
    return tuple(selected)
# end


def sequential_backward_elimination(xi: ndarray) -> tuple:
    """
    :param xi: set function
    :param s: n of features to select
    :return: set of selected features
    """
    p = len(xi)
    N = p - 1
    available = N    # iset([])
    removed = []

    while icount(available) > 0:
        f = -INF
        s = ihighbit(available)

        for i in imembers(available):
            S = iremove(available, i)
            if xi[S] > f:
                s = i
                f = xi[S]
            # end
        # end
        removed = removed + [s]
        available = iremove(available, s)
    # end
    return tuple(removed)
# end


# ---------------------------------------------------------------------------
# Set Function Feature Selection
# ---------------------------------------------------------------------------


class SFunFeatureSel:

    def __init__(self, sf: SetFunction=None):
        self.sf = sf
    # end

    def set(self, sf=None):
        self.sf = sf
        return self

    #
    # Sequential
    #

    def sequential_forward_selection(self) -> List[tuple]:
        xi = self.sf.xi
        return tosets(sequential_forward_selection(xi))
    # end

    def sequential_backward_elimination(self) -> List[tuple]:
        xi = self.sf.xi
        return tosets(sequential_backward_elimination(xi))
    # end

    #
    # Select sets
    #

    def best_set(self, k) -> tuple:
        return self.sf.best_set(k)

    def worst_set(self, k):
        return self.sf.worst_set(k)

    def best_sets(self) -> List[tuple]:
        return self.sf.best_sets()

    def worst_sets(self) -> List[tuple]:
        return self.sf.worst_sets()

    #
    # Power Indices
    #

    def banzhaf_values(self, k=None) -> List[tuple]:
        sf = self.sf
        if k is None:
            return tosets(sf.banzhaf_values(k=k))
        else:
            return sf.banzhaf_values()
    # end

    def shapley_values(self, k=None) -> List[tuple]:
        sf = self.sf
        if k is None:
            return tosets(sf.shapley_values(k=k))
        else:
            return sf.shapley_values()
    # end

    def chaining_values(self, k=None) -> List[tuple]:
        sf = self.sf
        if k is None:
            return tosets(sf.chaining_values(k=k))
        else:
            return sf.chaining_values()
    # end

    #
    # K-Power Indices
    #

    # def k_banzhaf_values(self) -> List[tuple]:
    #     n = self.sf.cardinality
    #     return [self.sf.banzhaf_values(k) for k in range(n+1)]
    # # end
    #
    # def k_shapley_values(self) -> List[tuple]:
    #     return self.sf.k_shapley_values()
    # # end
    #
    # def k_chaining_values(self) -> List[tuple]:
    #     return self.sf.k_chaining_values()
    # # end

# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
