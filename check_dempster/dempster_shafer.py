#
# Dempster-Shafer Theory
#
from typing import List, Tuple, Dict, Union, Iterable
from numpy import array, ndarray

from iset import iset, ilist, ipowersetn, isubsets, idiff, iunion, icard
from iset import imapset
from mathx import EPS
from sfun import MobiusTransform, SetFunction
from sfun.sfun_base import mobius_transform
from sfun.sfun_gen import zero_setfun, bayesian_mobius


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------

class MassFunction(MobiusTransform):

    @staticmethod
    def from_cardinality(n: int) -> "MassFunction":
        mt = zero_setfun(n)
        return MassFunction(mt)
    # end

    @staticmethod
    def from_bayesian(n: int) -> "MassFunction":
        mt = bayesian_mobius(n)
        return MassFunction(mt)

    @staticmethod
    def from_setfun(self: SetFunction) -> "MassFunction":
        xi = self.xi
        mt = mobius_transform(xi)
        return MassFunction(mt)
    # end

    def __init__(self, mt):
        super().__init__(mt)
    # end

    def _set(self, S: int, value: float) -> "MassFunction":
        N = self.length-1
        self.mt[S] = value
        self.mt[N] -= value
        return self
    # end

    def focal_sets(self, eps=EPS) -> List[Tuple[int]]:
        fs = []
        n = self.cardinality
        for S in ipowersetn(n):
            if self.mt[S] > eps:
                fs.append(ilist(S))
        return fs
    # end

    def focal_values(self, eps=EPS) -> Dict[Tuple[int], float]:
        fs = dict()
        n = self.cardinality
        for S in ipowersetn(n):
            v = self.mt[S]
            if v > eps:
                fs[ilist(S)] = v
        return fs
    # end

    def bel(self, A: Union[int, Iterable]) -> float:
        A = A if isinstance(A, int) else iset(A)
        return self._bel(A)

    def _bel(self, A: int) -> float:
        b = 0
        for S in isubsets(A):
            b += self.mt[S]
        return b

    def pl(self, A: Union[int, Iterable]) -> float:
        A = A if isinstance(A, int) else iset(A)
        return self._pl(A)

    def _pl(self, A: int) -> float:
        N = self.N
        cA = idiff(N, A)
        return 1 - self.bel(cA)

    def belief_function(self) -> "BeliefFunction":
        n = self.cardinality
        xi = zero_setfun(n)
        for S in ipowersetn(n):
            xi[S] = self._bel(S)
        return BeliefFunction(xi)
    # end

    def plausibility_function(self) -> "PlausibilityFunction":
        n = self.cardinality
        xi = zero_setfun(n)
        for S in ipowersetn(n):
            xi[S] = self._pl(S)
        return PlausibilityFunction(xi)
    # end
# end


class BeliefFunction(SetFunction):

    @staticmethod
    def from_pl(plf: "PlausibilityFunction"):
        pl = plf.data
        bf = 1 - pl
        return BeliefFunction(bf)

    def __init__(self, bf):
        super().__init__(bf)
    # end

    def plausibility_function(self):
        return PlausibilityFunction.from_bf(self)
    # end

    def mass_function(self) -> MassFunction:
        return MassFunction.from_setfun(self)
    # end
# end


class PlausibilityFunction(SetFunction):

    @staticmethod
    def from_bf(blf: "BeliefFunction"):
        bf = blf.data
        pl = 1 - bf
        return PlausibilityFunction(pl)

    def __init__(self, xi):
        super().__init__(xi)
    # end

    def belief_function(self):
        return BeliefFunction.from_pl(self)

    def mass_function(self) -> MassFunction:
        return self.belief_function().mass_function()
    # end
# end


class RefinementCoarseningTransformer:

    def __init__(self, c2r: Union[Tuple[int], List[int], ndarray], r: int = 0):
        """
        Coarse to refine mapping.
        It is necessary to passe, for each element of the coarse frame,
        the correspondent set of elements in the refined frame.

        This mapping is implemented using a list/ndarray, where at position i
        (that corresponds to the element 'i') it is assigned an integer representing
        the set in the refined frame

        :param c2r: coarse to refined set
        :param r: refined frame cardinality. If zero, it is induced from c2r
        """

        def c2r_card(c2r: Union[tuple, list, ndarray]):
            n = len(c2r)
            N = 0
            for i in range(n):
                N = iunion(N, c2r[i])
            return icard(N)

        self.c: int = len(c2r)
        self.r: int = r if r > 0 else c2r_card(c2r)
        self.c2r = array(c2r)
    # end

    def transform(self, mf: MassFunction) -> MassFunction:
        mfc = mf.cardinality
        if self.r == mfc:
            return self._coarsening(mf)
        if self.c == mfc:
            return self._refinement(mf)
        else:
            raise ValueError(f"Invalid MassFunction: cardinality {mfc} not in {{{self.c}, {self.r}}}")
    # end

    def _refinement(self, mf: MassFunction) -> MassFunction:
        rmf = MassFunction.from_cardinality(self.r)
        for i in range(self.c):
            S = self.c2r[i]
            m = mf[iset([i])]
            rmf.set(S, m)
        # end
        return rmf
    # end

    def _coarsening(self, mf: MassFunction) -> MassFunction:
        pass

# end
