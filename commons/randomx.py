from typing import Iterator
from sys import maxsize
from random import Random, random
from stdlib.imathx import comb, failing_fact

MAXINT = maxsize


def _normalize_k(k, n):
    """
        None    -> (0, n)
        int     -> (k, k)
        [k]     -> (0, k)
        (mi,ma) -> (mi, max)
    """
    if k is None:
        return (0, n)
    if isinstance(k, int):
        return (k, k)
    if len(k) == 1:
        return (0, k[0])
    else:
        return k
# end


def _normalize_plevels(n, k, plevels):
    """
    normalize plevels to be a list of n+1 elements, one for each level: 0...n

        None        -> proportional to the level's numerosity
        int/float   -> same probability for each level
        [l1..lk]     -> specified probability for the first k levels (the others will have 0)

    At the end sum(plevels) = 1

    :param k: (kmin, kmax) levels to use
    """
    kmin, kmax = k
    EPS = 1.e-6

    # normalize plevels length to be a list of n+1 elements
    if plevels is None:
        plevels = [comb(n, i) for i in range(n + 1)]
    elif isinstance(plevels, (int, float)):
        plevels = [plevels] * (n + 1)
    elif len(plevels) < n + 1:
        plevels = list(plevels) + [0] * (n + 1 - len(plevels))
    else:
        plevels = plevels[0:n + 1]

    assert len(plevels) == n + 1

    # clean the levels < kmin & > kmax
    for i in range(kmin): plevels[i] = 0
    for i in range(kmax + 1, n + 1): plevels[i] = 0

    # normalize the plevels to have sum(plevel) = 1
    total = sum(plevels)
    plevels = [plevels[i] / total for i in range(n + 1)]
    plevels[kmax] += 1. - sum(plevels) + EPS

    return plevels
# end


def _normalize_pitems(n, pitems):
    """
    normalize pitems to be a list of n elements
    """
    if pitems is None:
        pitems = [.5] * n
    elif type(pitems) in [int, float]:
        pitems = [pitems] * n
    elif len(pitems) < n:
        pitems = list(pitems) + [0] * (n - len(pitems))
    else:
        pitems = pitems[0:n]

    assert len(pitems) == n

    return pitems
# end


def _normalize_numerosity(n, k):
    """
    normalize the numerosity of the levels based on the selected levels k
    """
    kmin, kmax = k
    numerosity = [comb(n, i) for i in range(n + 1)]
    for k in range(kmin): numerosity[k] = 0
    for k in range(kmax + 1, n + 1): numerosity[k] = 0
    return numerosity
# end


# ---------------------------------------------------------------------------
# Random list
# ---------------------------------------------------------------------------

def random_list(n, normalized=False, grounded=False):
    """
    Generate a list of n random numbers
    """
    r = [random() for i in range(n)]
    if grounded:
        r[0] = 0.
    if normalized:
        s = sum(r)
        r = [r[i] / s for i in range(n)]
    return r
# end


# ---------------------------------------------------------------------------
# Random set generator
# ---------------------------------------------------------------------------

# WeightedRandomSets
#   Deve generare
#
#   1) set selezionati in modo uniforme (Banzhaf ...)
#   2) livelli selezionati in modo uniforme e poi set di quel livello selezionato in modo uniforme (Shapley)
#   3) livelli selezionati secondo una distribuzione di probabilita' e poi set di quel livello selezionato in modo
#      uniforme(Cardinal-Probabilistic)
#   4) set in cui gli elementi sono selezionati secondo una distribuzione di probabilita' (Player-Probabilistic).
#
#   Deve generare
#
#   1) set con cardinalita'(0, n)
#   2) set con cadinalita' (0, k)
#   3) set di cardinalita' esattamente (k, k).
#
class WeightedRandomSets:

    def __init__(self, n: int, m: int = None, k=None, plevels=None, pitems=None, random_state=None):
        """
        Generate a random subset from a set with n elements.
        If plevels is None and pitems is None

        :param n: n of elements in the full set
        :param m: n of sets to generate
        :param None|int|[int]|[int,int] k: cardinality of the set.
                None: generate a random set with cardinality [0, n]
                k: generate a random set with cardinality [k, k]
                [k]: generate a random set with cardinality [0,k]
                [kmin,kmax]: generate a random set with cardinality k in the range [kmin, kmax]
        :param None|number|list[number] plevels: probability assigned to each level
        :param None|number|list[number] pitems: probability assigned to each element
        :param random_state: seed for the local random generator
        """
        if m is None:
            m = MAXINT

        k = _normalize_k(k, n)

        # n of elements
        self.n = n
        """:type: int"""

        # (min,max) set cardinality
        self.k = k
        """:type: (int,int)"""

        # n of sets to generate
        self.m = m
        """:type: int"""

        # probability for each element of the set
        self.p_items = _normalize_pitems(n, pitems)
        """:type: list[float]"""

        # probability for each level
        self.p_levels = _normalize_plevels(n, k, plevels)
        """:type: list[float]"""

        # numerosity, n_sets
        self.numerosity = _normalize_numerosity(n, k)
        """:type: list[int]"""

        # max number of sets to generate
        self.n_sets = sum(self.numerosity)
        """:type: int"""

        # -------------------------------------------------------------------
        # Local implementation
        # -------------------------------------------------------------------

        # random state
        self.rnd = random_state if isinstance(random_state, Random) else Random(random_state)

        # next_set function
        # self._next_set = self._select_next_set_fun(k, plevels, pitems)

        # current element. Used to improve the random set generator
        self._elm = 0

        # n of generated sets in this session
        self._nsets = 0
    # end

    # -----------------------------------------------------------------------
    # Properties & Operations
    # -----------------------------------------------------------------------
    #   k == (kmin,kmax)
    #   numerosity
    #   n_sets
    #   p_levels
    #   p_items
    #

    def size(self):
        """Max number of sets that can generate"""
        return self.n_sets

    def random(self):
        """Next random set or None"""
        return self.next()

    def next(self):
        """Next random set or None"""
        s = self._next_set()
        return None if s is None else set(s)

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    def _next_set(self):
        if self._nsets > self.m:
            return None

        k = self._cardinality()
        s = self._rnd_set(k)
        self._nsets += 1
        return s

    def _rnd_set(self, k):
        rnd = self.rnd
        e = self._elm
        n = self.n
        s = set()
        while len(s) != k:
            r = rnd.random()
            if r <= self.p_items[e]:
                s.add(e)
            e = (e + 1) % n

        self._elm = e
        self._nsets += 1
        return set(s)

    def _cardinality(self):
        n = self.n
        r = self.rnd.random()
        cdf = 0.
        for k in range(n + 1):
            cdf += self.p_levels[k]
            if r <= cdf:
                return k
        return n

    # -----------------------------------------------------------------------
    # Iterator
    # -----------------------------------------------------------------------

    def __iter__(self) -> Iterator[set]:
        return self

    def __next__(self) -> set:
        s = self.next()
        if s is None:
            raise StopIteration
        else:
            return s
# end


class RandomSets:

    def __init__(self, n: int, m: int = None, k: int = None, unique=True, random_state=None, generated=None):
        """
        Generate a random subset from a set with n elements

        :param n: n of elements in the full set
        :paramm: n of sets to generate
        :param None|int|[k]|[int,int] k: cardinality of the set.
                None: generate a random set
                k: generate a random set with cardinality k
                (kmin,kmax): generate a random set with cardinality k in the specified range
        :param unique: if to generate permutations without duplication
        :param random_state: seed used for the local random generator
        :param generated: set of already generated subsets (IT IS UPDATE)
        """

        if m is None:
            m = MAXINT

        k = _normalize_k(k, n)

        # n of elements
        self.n = n
        """:type: int"""

        # (min,max) set cardinality
        self.k = k
        """:type: (int,int)"""

        # n of sets to generate
        self.m = m
        """:type: int"""

        # if to generate unique sets
        self.unique = unique
        """:type: bool"""

        # numerisity of the levels
        self.numerosity = _normalize_numerosity(n, k)
        """:type: list[int]"""

        # max number of sets to generate
        self.n_sets = sum(self.numerosity)
        """:type: int"""

        # -------------------------------------------------------------------
        # Local implementation
        # -------------------------------------------------------------------

        # local random generator
        self.rnd = random_state if isinstance(random_state, Random) else Random(random_state)

        # already generated sets as bigint
        self.generated = set() if generated is None else generated

        # last member generated
        self._elm = n - 1

        # n of sets generated in this session
        self._nsets = 0
    # end

    # -----------------------------------------------------------------------
    # Properties & Operations
    # -----------------------------------------------------------------------

    def size(self) -> int:
        """Max number of sets that can generate"""
        return self.n_sets

    def random(self) -> set:
        """Next random set or None"""
        return self.next()

    def next(self) -> set:
        """Next random set or None"""
        s = self._next_set()
        return None if s is None else set(s)

    @property
    def p_levels(self):
        n = self.n
        k = self.k
        return _normalize_plevels(n, k)

    @property
    def p_items(self):
        n = self.n
        return _normalize_pitems(m, None)

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    def _next_set(self):
        def spack(l: list) -> int:
            """convert a set of itegers in a bigint"""
            return sum(1 << i for i in l)

        s = None
        retry = True
        itry = 0
        while retry and len(self.generated) < self.n_sets and \
                self._nsets < self.m and \
                itry <= len(self.generated):
            k = self._cardinality()
            s = self._rnd_set(k)
            t = spack(s)
            retry = self.unique and t in self.generated
            if retry:
                s = None
                itry += 1
                continue
            if self.unique:
                self.generated.add(t)
            self._nsets += 1
        return s
    # end

    def _rnd_set(self, k):
        """Generate a random set of cardinality k"""
        n = self.n
        s = set()
        while len(s) != k:
            self._elm = (self._elm + 1) % n
            if self.rnd.random() > .5:
                s.add(self._elm)
            # end
        # end
        return sorted(s)
    # end

    def _cardinality(self):
        """Generate a cardinality in the range [kmin,kmax], based on the
        numerosity of each level"""
        r = self.rnd.random()
        kmin, kmax = self.k
        c = 0
        n = self.n_sets
        for k in range(kmin, kmax + 1):
            c += self.numerosity[k]
            if r <= c / n:
                return k
        return kmax
    # end

    # -----------------------------------------------------------------------
    # Iterator
    # -----------------------------------------------------------------------

    def __iter__(self) -> Iterator[set]:
        return self

    def __next__(self) -> set:
        s = self.next()
        if s is None:
            raise StopIteration
        else:
            return s
# end


class RandomPermutations:

    def __init__(self, n, m=None, k=None, unique=True, random_state=None, generated=None):
        """
        :param n: number of elements in the permutation
        :param m: number of permutations to generate
        :param int|(int,int) k: size of the permutation
        :param unique: if to generate sets without duplication
        :param random_state: initilize the random generator
        """
        if m is None:
            m = MAXINT
        if k is None:
            k = n

        # n of elements
        self.n = n
        """:type: int"""

        # permutations cardinality
        self.k = k
        """:type: int"""

        # n of permutations to generate
        self.m = m
        """:type: int"""

        # if to generate unique permutations
        self.unique = unique
        """:type: bool"""

        # max number of sets to generate
        self.n_perms = failing_fact(n, k)
        """:type: int"""

        # -------------------------------------------------------------------
        # Local implementation
        # -------------------------------------------------------------------

        # local random generator
        self.rnd = random_state if isinstance(random_state, Random) else Random(random_state)

        # already generated permutations
        self.generated = set() if generated is None else generated

        # n of permutations generated in this session
        self._nperm = 0
        """:type: int"""

        # last permutation
        _self._perm = list(range(n))
        """:type: list"""
    # end

    # -----------------------------------------------------------------------
    # Properties & Operations
    # -----------------------------------------------------------------------

    def size(self) -> int:
        return self.n_try
    # end

    def random(self) -> tuple:
        return self.next()
    # end

    def next(self) -> tuple:
        p = self._next_perm()
        return None if p is None else tuple(p)
    # end

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------

    def _next_perm(self):
        def ppack(l: list) -> tuple:
            return tuple(l)

        k = self.k
        m = self.m
        p = None
        retry = True
        itry = 0
        while retry and len(self.generated) < self.n_perms and \
                self._nperms < self.m and \
                itry <= len(self.generated):
            p = self._rnd_perm(k)
            t = ppack(p)
            retry = self.unique and t in self.generated
            if retry:
                p = None
                itry += 1
                continue
            if self.unique:
                self.generated.add(t)
            self._nperm += 1
        return p
    # end

    def _rnd_perm(self, k):
        self.rnd.shuffle(self._perm)
        return self._perm[0:k]
    # end

    # -----------------------------------------------------------------------
    # Iterator
    # -----------------------------------------------------------------------

    def __iter__(self) -> Iterator[tuple]:
        return self

    def __next__(self) -> tuple:
        p = self.next()
        if p is None:
            raise StopIteration
        else:
            return p
    # end
# end


# ---------------------------------------------------------------------------
# SetsStatistics
# ---------------------------------------------------------------------------
# Used to analyze the statistics of the set generators
# Keep the following informations
#
#   - n of sets with a specific cardinality
#   - n of elements in the sets
#

class SetsStatistics:

    def __init__(self, n=0, prec=3, estats=False):
        """
        Compute the set statistics
        :param n: n of elements of the set
        :param bool estats: if to check the elements statistics
        """
        self.n = n
        self.k = (0, n)
        self.N = set(range(n))
        self.c_levels = [0] * (n + 1)  # levels used
        self.c_in_set = [0] * n  # e is IN set
        self.c_outset = [0] * n  # e is NOT in set
        self.k = [n, 0]  # selected levels
        self.p = prec

        nsets = 1 << n
        self.numerosity = [comb(n, k) for k in range(n + 1)]
        self.p_levels = [self.numerosity[k] / nsets for k in range(n + 1)]
        self.p_items = [.5 for i in range(n)]
    # end

    def set(self, wrs: WeightedRandomSets):
        n = wrs.n
        self.n = wrs.n
        self.k = list(wrs.k)
        self.N = set(range(n))
        self.c_levels = [0] * (n + 1)  # levels used
        self.c_in_set = [0] * n  # e is IN set

        self.numerosity = wrs.numerosity
        self.p_levels = wrs.p_levels
        self.p_items = wrs.p_items
        return self
    # end

    def add(self, S):
        # S = set(S)
        s = len(S)
        self.c_levels[s] += 1

        self.k[0] = min(self.k[0], s)
        self.k[1] = max(self.k[1], s)

        for e in S:
            self.c_in_set[e] += 1
    # end

    def check(self, wrs: WeightedRandomSets):
        self.set(wrs)
        for S in wrs:
            self.add(S)
        self.report()
    # end

    def report(self):
        p = self.p
        n = self.n
        k = self.k
        n_sets = sum(self.c_levels)

        print("Set of %d elements" % self.n)
        print("Levels:")
        print("  used", self.c_levels)
        print("     %", [round(self.c_levels[i] / n_sets, p) for i in range(n + 1)])
        print("   def", [round(self.p_levels[k], p) for k in range(n + 1)])

        print("Elements:")
        print("  used", self.c_in_set)
        print("     %", [round(self.c_in_set[i] / n_sets, p) for i in range(n)])
        print("   def", [self.p_items[k] for k in range(n)])
        print("End")
    # end
# end
