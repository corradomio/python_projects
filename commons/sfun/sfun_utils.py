#
# SetFunction utilities
#
#   A union B       = A + B
#   A intersect B   = A * B
#   A difference B  = A - B
#   complement A    = !A = X - A
#
#   additive:       xi[A + B]  = xi[A] + xi[B]
#   subadditive:    xi[A + B] <= xi[A] + xi[B]
#   superadditive:  xi[A + B] >= xi[A] + xi[B]
#
#   modular:        xi[A + B] + xi[A * B]  = xi[A] + xi[B]
#   submodular:     xi[A + B] + xi[A * B] <= xi[A] + xi[B]
#   supermodular:   xi[A + B] + xi[A * B] >= xi[A] + xi[B]
#
#   (sub|super)modular -> (sub|super)additive
#       reverse NOT TRUE
#
#   monotone:       xi[A] <= xi[B]  if A <= B
#   grounded:       xi[0] = 0
#   normalized:     xi[X] = 1
#   conjugate:      xc[A] = xi[X] - xi[X-A]
#                   xc[X] = xi[X] - xi[X-X] = xi[X].
#
from random import shuffle
from imathx import ipow
from .sfun_fun import *


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _from_dict(data: dict):
    """
    Set function defined by dictionary {subset: value, ...}:

        key:    subset (as tuple or int)
        value:  function value

    Note: the subsets are composed by the integers 0,1,...

    :param data:
    :return:
    """
    # def _min(x, l):
    #     return x if len(l) == 0 else min(x, min(l))

    def _max(x, l):
        return x if len(l) == 0 else max(x, max(l))

    t = 0
    for u in data:
        t = _max(t, u)
    # end
    s = 0

    n = t - s + 1
    p = 1 << n
    xi = fzeros(p)
    for u in data:
        a = [i-s for i in u]
        S = iset(a)
        xi[S] = data[u]
    return xi
# end


def _from_list(data: list):
    """
    Set function defined by a list [subset, value, subset, ...]

    Note: the subsets are composed by the integers 0,1,...

    :param data:
    :return:
    """
    # def _min(x, l):
    #     return x if len(l) == 0 else min(x, min(l))

    def _max(x, l):
        return x if len(l) == 0 else max(x, max(l))

    t = 0
    for k in range(0, len(data), 2):
        u = data[k]
        t = _max(t, u)
    # end
    s = 0

    n = t - s + 1
    p = 1 << n
    xi = fzeros(p)
    for t in range(0, len(data), 2):
        u = data[t]
        a = [i - s for i in u]
        S = iset(a)
        xi[S] = data[t + 1]
    return xi
# end


# ---------------------------------------------------------------------------
# Bases
# ---------------------------------------------------------------------------


# base for Shapley
def base_shapley(S, T):
    def beta(k, l):
        return sum(comb(k, j) * bernoulli(l - j) for j in range(k + 1))

    s = icard(S)
    i = icard(iinterset(S, T))
    return beta(i, s)
# end


# base for Banzhaf
def base_banzhaf(S, T):
    t = icard(T)
    return ipow(1/2, t)*idsign(S, T)
# end


# ---------------------------------------------------------------------------
# Player weights
# ---------------------------------------------------------------------------

def player_weight(p:ndarray, S: int) -> float:
    n = len(p)
    N = isetn(n)
    D = idiff(N, S)

    w = 1.
    for i in imembers(S):
        w *= p[i]
    for i in imembers(D):
        w *= (1-p[i])
    return w


def player_level(p, k:int) -> float:
    n = len(p)
    N = isetn(n)
    k = parse_k(k, n, 0)

    l = 0.
    for S in ilexsubset(N, k=k):
        l += player_weight(p, S)
    return l


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------

def _minimum_subsets(n, c):
    """
    Generate the list of all subsets with cardinality 1 and 2, based on the value
    of c (number of subsets to generate):

        c >= n -> generate all subsets of cardinality 1
        c >= n+n*(n-1)/2 -> generate all subsets of cardinality 2

    Add always the full set

    """
    subsets = set()
    p = 1 << n
    if c > p: c = n

    # add the fullset
    subsets.add(p-1)

    # ensure to add all subsets of 1 element
    if c >= n:
        for i in range(n):
            subsets.add(iset(i))

    # add all subsets of two elements
    if c >= (n + n * (n - 1) // 2):
        for i in range(n):
            for j in range(i + 1, n):
                subsets.add(iset([i, j]))

    return subsets
# end


def subsets_from_permutations(n, c):
    """
    Generates a random permutation and from this permutation
    generates the subsets of 0,1,2,...n elements

    :param n: n of elements of the set
    :param c: n of sets to generate
    :return: list of subsets
    """
    p = 1 << n
    if c > p: c = n

    subsets = _minimum_subsets(n, c)

    # add the subsets using permutations
    # note: the fullset is already added
    P = list(range(n))
    while len(subsets) < c:
        for k in range(1, n):
            S = iset(P[0:k])

            subsets.add(S)
        shuffle(P)
    return list(subsets)
# end


def random_subsets(n, c):
    """
    Generate random subsets

    :param n: n of elements in the set
    :param c: n of sets to generate
    :return: a list of subsets
    """
    from random import randrange
    subsets = _minimum_subsets(n, c)
    p = (1 << n) - 1
    if c > p: c = p

    # add the extra subsets
    while len(subsets) < c:
        subsets.add(randrange(0, p))
    return list(subsets)
# end


def subsets_from_steps(n, c):
    """
    Generate a list of subsets from tha integer interval [0, 2^p) with steps of 2^p/c
    Note: the number of sets ig, in general, greater than 'c'

    :param p: n of elements in the set
    :param c: n of subsets to generate (but in the result will be a grated number of sets)
    :return: list of subsets
    """
    subsets = _minimum_subsets(n, c)
    p = 1 << n

    # add the sets based on the integers
    s = p//c
    for i in range(0, p, s):
        subsets.add(i)

    return subsets
# end


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------

class Lattice:

    def __init__(self, n: int):
        """
        Lattice of subset relations for a set with 'p' elements.

        The elements are the integers {0,1,...p-1}

        :param n: number of elements in the set
        """
        p = 1 << n
        N = p - 1
        q = n * (n - 1) // 2
        self.p = n
        self.p = p
        self.N = N
        self.q = q
    # end

    def rand_subsets(self, n, mode="permutation"):
        """
        Return a list of subsets generated in the following way:

            "perm":
                - generate 'n' permutations of the list [0,1,...p)
                - for each permutation, generate the subsets based on the prefix
                - collect all subsets

            "set"
                - generate 'n' random sets

            "step"
                - generate 'n' subsets using the integers [...k*step...]

        The fullset is always added.
        The sets with 1 and 2 elements are added if

            - 1 element sets added if n > p
            - 2 element sets added if n > p*(p-1)/2

        Note: altre strategie di campionamento potrebbero essere selezionate in base, ad esempio,
              al numero di elementi per ogni livello

        :param n: (int | float) n of permutations
        :return: list of subsets (in bitset format)
        """
        p = self.p

        if isinstance(n, float):
            n = int(n * self.p)

        if mode.startswith("perm"):
            return subsets_from_permutations(p, n)
        elif mode.startswith("set"):
            return random_subsets(p, n)
        elif mode.startswith("step"):
            return subsets_from_steps(p, n)
        else:
            raise ValueError("Unsupported mode '{}'".format(mode))
    # end

    def subsets(self) -> Iterator[int]:
        """
        Iterator on all subsets in the lattice

        :return: subsets (int bitset format) iterator
        """
        N = self.N
        for S in ipowerset(N):
            yield S
    # end

    #
    # Experimental
    #

    def coords(self, S: int):
        p = self.p
        n = self.p
        y = icard(S)/p
        x = S/(n-1)
        return x, y
    # end

    def surface(self, sf: SFun) -> tuple:
        xs = []
        ys = []
        zs = []

        for S in self.subsets():
            x, y = self.coords(S)
            z = sf.eval(S)

            xs.append(x)
            ys.append(y)
            zs.append(z)
        return xs, ys, zs
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
