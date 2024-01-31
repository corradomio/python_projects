#
#
#
from random import random, gauss, Random

from numpy import sort, flip, array, ndarray, zeros
from numpy.random import rand

from iset import *
from stdlib.mathx import comb, iseven
from .sfun_base import fzeros
from .sfun_base import inverse_mobius_transform, conjugate


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def random_value(n: int,
                 grounded: bool=False,
                 normalized: bool=False,
                 pdistrib: bool=False,
                 sorted: bool=False,
                 rnd=None) -> ndarray:
    """
    Return a random vector of n elements

    :param grounded: v[0] = 0
    :param normalized: max(v) = 1
    :param pdistrib: sum(v) = 1
    :param sorted: sorted(v)
    """
    if rnd is None:
        rnd = Random()

    v = array([rnd.random() for i in range(n)])

    if grounded:
        v[0] = 0.
    if normalized:
        v /= v.max()
    if pdistrib:
        v /= v.sum()
    if sorted:
        v.sort()
        v = flip(v)

    return v
# end


def random_op(n: int,
              prob_interfere: float,
              dim: int = 2,
              distrib="discrete",
              rnd=None) -> ndarray:
    """
    Generate the table of operators '+' (+1) and '-' (-1), or a table of factors
    in the range [-1, +1]

    :param n: n of elements in the set
    :param prob_interfere: probability of interference ('-' operator)
    :param dim: dimensions of the
    :param distrib: distribution probability ("discrete", "uniform"). Default: "discrete" ({-1,+1})

    :return:
    """
    def interfere_factor():
        if distrib == "discrete":
            return -1 if rnd.random() < prob_interfere else +1
        elif distrib == "uniform":
            return -1 + 2 * rnd.random()

    if rnd is None: rnd = Random()

    if dim == 1:
        op = zeros(n, dtype=float)
        for i in range(n):
            op[i] = interfere_factor()
    elif dim == 2:
        op = zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                op[i, j] = op[j, i] = interfere_factor()
        op[n - 1, 0] = interfere_factor()
    else:
        raise ValueError("Unsupported dimensions %d".format(dim))
    return op
# end


def eval_linear(S, v, op):
    """
    Evaluate a set function where the value of the set S is composed as:

        vS = sum[ v[i]+op[j]*v[j]  i=0..n-1, j=(i+1)%n ]

    :param S: set (in bit format)
    :param v: scores
    :param op: operators op[i]
    :return: value of the set
    """
    if isinstance(S, int):
        S = ilist(S)

    s = len(S)
    if s == 0:
        return 0.
    if s == 1:
        i = S[0]
        return v[i]

    vS = 0.
    c = 0.
    for a in range(s):
        b = (a+1) % s
        i = S[a]
        j = S[b]
        vS += abs(v[i] + op[j] * v[j])
        c += 1
    # end
    return vS/c
# end


def eval_digraph(S, v, op):
    """
    Evaluate a set function
    :param S: set (in bit format)
    :param v: scores
    :param op: operators op[i,j]
    :return: value of the set
    """
    if isinstance(S, int):
        S = ilist(S)
        
    s = len(S)
    if s == 0:
        return 0.
    if s == 1:
        i = S[0]
        return v[i]

    vS = 0.
    c = 0
    for a in range(0, s-1):
        i = S[a]
        for b in range(a+1, s):
            j = S[b]
            vS += abs(v[i] + op[i, j]*v[j])
            c += 1
        # end
    # end
    return vS/c
# end


# ---------------------------------------------------------------------------
# simplex
# ---------------------------------------------------------------------------

def simplex(n: int, grounded=False) -> ndarray:
    """
    Generate a sequence of numbers such that

        sum(s) = 1
        s[i] >= 0

    :param n: n of number
    :param grounded: if True, s[0] must be 0 (zero)
    :return: random sequence
    """
    s = fzeros(n)
    a = rand(n)     # are used ONLY n-1 values
    a[0] = 0
    a = sort(a)
    if grounded:
        s[1:n-1] = a[1:n-1] - a[0:n-2]
    else:
        s[0:n-1] = a[1:n] - a[0:n-1]
    s[n-1] = 1 - s.sum()
    return s
# end


# ---------------------------------------------------------------------------
# Set function generators
# ---------------------------------------------------------------------------
# Note: the function's value on the fullset is 1.
#

def normalize(xi: ndarray, mode=False):
    """
    Normalize the array of values.
    If mode is:

        False:  sum(s) == 1
        True:   max(s) == 1
        None:   none

    :param xi:
    :param mode:
    :return:
    """
    if mode is None:
        return xi
    if mode:
        m = xi.max()
    else:
        m = xi.sum()
    return xi / m
# end


def rand_simplex_mobius(n: int) -> ndarray:
    """
    Generate a sequence of 2^p random values such as:

        xi[0]   = 0
        sum(xi) = 1
        xi[i]  >= 0  i = [1 .. 2^p)

    :param n: number of elements of the set
    :return:  array of subset values
    """
    p = 1 << n
    m = simplex(p, grounded=True)
    return m
# end


def rand_normal_mobius(n: int, mode) -> ndarray:
    p = 1 << n
    N = p - 1
    m = fzeros(p)

    negate_even = True if mode == "subn" else False

    for S in ipowerset(N, empty=False):
        s = icard(S)
        mS = gauss(0., 1.)

        if negate_even and iseven(s):
            mS = -mS

        m[S] = mS
    # return m/sum(m)
    return m
# end


def rand_bounded_mobius(n: int) -> ndarray:
    p = 1 << n
    N = p - 1
    m = fzeros(p)

    for S in ipowerset(N, empty=False):
        s = icard(S)

        la = 2*(s//4)
        lp = 2*((s-1)//4)+1

        l = -comb(n-1, lp)
        r = +comb(n-1, la)

        m[S] = l + (r-l)*random()
    # return m/sum(m)
    return m
# end


def rand_superadd_mobius(n: int) -> ndarray:
    p = 1 << n
    N = p - 1
    m = fzeros(p)

    for S in ipowerset(N, empty=False):
        s = icard(S)

        la = 2*(s//4)
        lp = 2*((s-1)//4)+1

        l = 0 # -comb(n-1, lp)
        r = +comb(n-1, la)

        m[S] = l + (r-l)*random()
    # return m/sum(m)
    return m
# end


def rand_subadd_mobius(n: int) -> ndarray:
    p = 1 << n
    N = p - 1
    m = fzeros(p)

    for S in ipowerset(N, empty=False):
        s = icard(S)

        la = 2*(s//4)
        lp = 2*((s-1)//4)+1

        if iseven(s):
            l = -comb(n-1, lp)
            r = 0
        else:
            l = 0
            r = +comb(n-1, la)

        m[S] = l + (r-l)*random()
    # return m/sum(m)
    return m
# end


def rand_mobius(n: int) -> ndarray:
    p = 1 << n
    m = rand(p)
    # return m/sum(m)
    return m
# end


# ---------------------------------------------------------------------------

def compose_monotone(p, empty=True):
    N = p - 1  # full set
    xi = fzeros(p)

    for S in isubsets(N, lower=empty):
        s = 0.
        for T in isubsets(S, lower=empty, upper=False):
            s = max(s, xi[T])
        xi[S] = s + random()
    return xi
# end


def rand_monotone(n: int, grounded=True, mode=None) -> ndarray:
    """
    Generate a monotone function:

        xi(A) <= xi(B)  if A subset B

    :param n: number of elements of the set
    :param grounded: if xi({}) must be 0
    :return:  array of subset values
    """
    p = 1 << n              # n of elements
    xi = compose_monotone(p, not grounded)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_additive(r: ndarray) -> ndarray:
    n = len(r)
    p = 1 << n
    N = p - 1
    xi = fzeros(p)

    for i in range(n):
        S = iset(i)
        xi[S] = r[i]

    for A, B in isubsetpairs(N):
        U = iunion(A, B)
        assert iinterset(A, B) == 0
        xi[U] = xi[A] + xi[B]
    return xi
# end


def rand_additive(p: int, mode=None) -> ndarray:
    """
    Generate an additive function

    :param p: n of elements in the set
    :return:  array of subset values
    """
    r = rand(p)
    xi = compose_additive(r)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_superadditive(r: ndarray) -> ndarray:
    n = len(r)
    p = 1 << n
    m = simplex(p, grounded=True)
    xi = inverse_mobius_transform(m)
    return xi
# end


def rand_superadditive(n: int, mode=None) -> ndarray:
    """
    Generate a super additive function

    Note: a 'superadditive' function is also a 'monotone' function

    :param n: n of elements in the set
    :return:  array of subset values
    """
    r = rand(n)
    xi = compose_superadditive(r)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_subadditive(r: ndarray) -> ndarray:
    xi = compose_superadditive(r)
    xi = conjugate(xi)
    return xi
# end


def rand_subadditive(n: int, mode=None) -> ndarray:
    """
    Generate a super additive function

    Note: a 'superadditive' function is also a 'monotone' function

    :param n: n of elements in the set
    :return:  array of subset values
    """
    r = rand(n)
    xi = compose_subadditive(r)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_ladditive(r: ndarray, l: float) -> ndarray:
    """
    Compose a lambda-measure

        xi(A union B) = xi(A) = xi(B) + lambda*xi(A)*xi(B)

    :param r: values of the elements in the set
    :param l:
    :return:
    """
    n = len(r)
    p = 1 << n
    N = p - 1
    xi = fzeros(p)

    for i in range(n):
        # S = iset(i)
        S = i
        xi[S] = r[i]

    for A, B in isubsetpairs(N):
        U = iunion(A, B)
        if xi[U] == 0:
            xi[U] = xi[A] + xi[B] + l*xi[A]*xi[B]
    return xi
# end


def rand_ladditive(n: int, l: float, mode=None) -> ndarray:
    """
    Generate a lambda-measure

    :param n: n of elements in the set
    :param l: lambda value
    :return:  array of subset values
    """
    r = rand(n)
    xi = compose_ladditive(r, l)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_modular(r: ndarray) -> ndarray:
    """
    Compose a modular function

        xi(A + B) + xi(A * B) = xi(A) + xi(B)

        xi(A + B) = xi(A) + xi(B) - xi(A * B)

    :param r:
    :return:
    """
    n = len(r)
    p = 1 << n
    N = p - 1
    xi = fzeros(p)  # subset values

    for i in range(n):
        S = iset(i)
        xi[S] = r[i]

    for A, B in isubsetpairs(N):
        U = iunion(A, B)
        I = iinterset(A, B)

        xi[U] = xi[A] + xi[B] - xi[I]
    return xi
# end


def rand_modular(n: int, mode=None) -> ndarray:
    r = rand(n)
    xi = compose_modular(r)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_supermodular(r: ndarray) -> ndarray:
    """
    Compose a supermodular function

        xi[A + B] + xi[A * B] >= xi[A] + xi[B]

        xi[A + B] >= xi[A] + xi[B] - xi[A * B]

    :param r:
    :return:
    """
    n = len(r)
    p = 1 << n
    N = p - 1
    xi = fzeros(p)  # subset values

    def rand(S):
        # r = 1.
        # for i in range(icard(S)):
        #     r *= random()
        # return r
        return random()

    for i in range(n):
        S = iset(i)
        xi[S] = r[i]

    for A, B in isubsetpairs(N):
        U = iunion(A, B)
        I = iinterset(A, B)

        if xi[U] <= (xi[A] + xi[B] - xi[I]):
            xi[U] = (xi[A] + xi[B] - xi[I]) + rand(U)*xi[A]*xi[B]*(1 - xi[I])
    return xi
# end


def rand_supermodular(n: int, mode=None) -> ndarray:
    r = rand(n)
    xi = compose_supermodular(r)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def compose_submodular(r: ndarray) -> ndarray:
    """
    Compose a supermodular function

        xi[A + B] + xi[A * B] <= xi[A] + xi[B]

        xi[A + B] <= xi[A] + xi[B] - xi[A * B]

    :param r:
    :return:
    """
    n = len(r)
    p = 1 << n
    N = p - 1
    xi = fzeros(p)  # subset values

    def _max(S):
        return max((xi[T] for T in isubsets(S, upper=False)), default=0)

    def _min(S):
        return min((xi[T] for T in isubsets(S, upper=False)), default=0)

    for i in range(n):
        S = iset(i)
        xi[S] = r[i]

    for A, B in isubsetpairs(N):
        U = iunion(A, B)
        I = iinterset(A, B)

        if xi[U] == 0 or xi[U] > (xi[A] + xi[B] - xi[I]):
            mu = _min(U)
            mi = _max(I)
            xi[U] = mu + random()*max(xi[A] + xi[B] - mi, 0)
    return xi
# end


def rand_submodular(n: int, mode=None) -> ndarray:
    r = rand(n)
    xi = compose_submodular(r)
    xi = normalize(xi, mode=mode)
    return xi
# end


# ---------------------------------------------------------------------------

def zero_setfun(n: int) -> ndarray:
    """
    Generate a function with all zeros, except xi[N] = 1

    :param n:
    :return:
    """
    p = 1 << n
    xi = fzeros(p)
    xi[p - 1] = 1.
    return xi
# end


def bayesian_setfun(n: int) -> ndarray:
    """
    Generate a function based on probabilties assigned to singletons,
    and all other sets have probability equals to sum(P(i) i in S)

    :param n:
    :return:
    """
    p = 1 << n
    xi = fzeros(p)

    # generate n random values with sum=1
    r = [random() for i in range(n)]
    s = sum(r)
    for S in ipowersetn(n):
        xi[S] = sum(r[e]/s for e in imembers(S))
    xi[p - 1] = 1.
    return xi
# end


def rand_setfun(n: int) -> ndarray:
    """
    Generate a random function with the properties:

        xi[{}]      = 0.
        xi[N]       = 1.

    :param n: n of elements in the set
    :return:  array of subset values
    """
    p = 1 << n
    xi = rand(p)
    xi[0] = 0.
    xi[p - 1] = 1.
    return xi
# end


def const_setfun(n: int) -> ndarray:
    """
    Generate a constant set function with the properties:

        xi[{}]      = 0
        xi[A]       = 1/(2^p-1)

    :param n: n of elements in the set
    :return:  array of subset values
    """
    p = 1 << n          # number of subsets
    xi = fzeros(p)       # subset values
    xi[1:] = 1/(p-1)
    return xi
# end


def rand_weigthed_mobius(wl, wh, mode=None) -> ndarray:
    """
    :param wl: low weight
    :param wh: high weight
    :return:
    """
    n = len(wl)
    p = 1 << n
    m = fzeros(p)

    for S in ipowersetn(n):
        s = icard(S)-1
        m[S] = wl[s] + (wh[s]-wl[s])*random()
    m = normalize(m, mode=mode)
    return m
# end


# ---------------------------------------------------------------------------

def bayesian_mobius(n: int) -> ndarray:
    """
        Generate a function based on probabilties assigned to singletons,
        and all other sets have probability equals to sum(P(i) i in S)

        :param n:
        :return:
        """
    p = 1 << n
    mt = fzeros(p)

    r = [random() for i in range(n)]
    s = sum(r)
    for i in range(n):
        S = 1 << i
        mt[S] = r[i]/s
    return mt
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
