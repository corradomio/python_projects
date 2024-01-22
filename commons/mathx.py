from math import *
from typing import List, Union, Iterable
from stdlib import sq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INF: float = float('inf')


# ---------------------------------------------------------------------------
# chop
# ---------------------------------------------------------------------------

def chop(x: Union[float, list[float]], l: float, u: float) -> Union[float, list[float]]:
    """Clip x in range [l,u] """
    if isinstance(x, (int, float)):
        if x < l: return l
        if x > u: return u
        return x
    else:
        return [chop(e, l, u) for e in x]
# end


# ---------------------------------------------------------------------------
# argsort
# ---------------------------------------------------------------------------

# def argsort(values: Iterable, descending: bool = False) -> List[int]:
#     """Sort the values in ascending (ore descending) order and return the indices"""
#     n = len(list(values))
#     pairs = [(i, values[i]) for i in range(n)]
#     pairs = sorted(pairs, key=lambda p: p[1], reverse=descending)
#     return [p[0] for p in pairs]
# # end


# ---------------------------------------------------------------------------
# Real comparisons with error
# ---------------------------------------------------------------------------
# The following comparison predicates can be used for float values where it
# is not possible to do 'safe' comparisons without consider rounding/accumulating
# errors. In this case, it is possible to specify an 'eps' (that can be 'zero')
# to absorb these errors.

EPS: float = 1.e-6


def sign(x, zero=False, eps: float = EPS) -> int:
    """
    Sign of the number:

        -1 if in range (-inf, -eps)
         0 if in range [-eps, +eps]
        +1 if in range (+eps, +inf)

    Note that 'eps' can be 0, in this case 'sign' is 0 only for exactly 0 (zero)

    :param x: value to analyze
    :param zero: if to return 0 (True) or 1 (False) for 'zero values'
    :param eps: values interval to consider 'zero'
    :return: the integer values -1, 0, 1, based on 'x' value
    """
    if x < -eps: return -1
    if x > +eps: return +1
    return 0 if zero else 1


def zero(x, eps: float = EPS) -> float:
    """return 0 if the value is smaller than an eps"""
    return 0. if -eps <= x <= +eps else x


def isz(x: float, eps: float = EPS) -> bool:
    """is zero"""
    return -eps <= x <= eps


def isnz(x: float, eps: float = EPS) -> bool:
    """is not zero"""
    return not isz(x, eps=eps)


def iseq(x: float, y: float, eps: float = EPS) -> bool:
    """is equal to"""
    return isz(x - y, eps=eps)


def isgt(x: float, y: float, eps: float = EPS) -> bool:
    """is greater than"""
    return x > (y + eps)


def islt(x: float, y: float, eps: float = EPS) -> bool:
    """is less than"""
    return x < (y - eps)


def isge(x: float, y: float, eps: float = EPS) -> bool:
    """is greater or equal than"""
    return not islt(x, y, eps=EPS)


def isle(x: float, y: float, eps: float = EPS) -> bool:
    """is less or equal than"""
    return not isgt(x, y, eps=EPS)


# ---------------------------------------------------------------------------

def iseven(n: int) -> bool:
    """e' pari"""
    return n % 2 == 0  # e' pari


def isodd(n: int) -> bool:
    """e' dispari"""
    return n % 2 == 1  # e' dispari


# ---------------------------------------------------------------------------
# Extra math functions
# ---------------------------------------------------------------------------

# inverse with check
def inv(x: float, eps: float = EPS) -> float:
    """
    Inverse of the number with check for zero.
    If x is zero, return zero

    :param x: value
    :param eps: epserance
    :return: 1/x or 0
    """
    return 0. if isz(x, eps=eps) else 1. / x


# Square
def sq(x: float) -> float: return x * x


# ---------------------------------------------------------------------------
# mean, var, standard deviation
# ---------------------------------------------------------------------------

def mean(iterable: Iterable[float], default: float = 0.) -> float:
    """mean of a collection"""
    s = 0.
    n = 0
    for v in iterable:
        s += v
        n += 1
    # n = len(l)
    # return sum(l)/n if n > 0 else 0.
    return s / n if n > 0 else default


def var(iterable: Iterable[float], m=None) -> float:
    """variance of a collection"""
    if m is None:
        m = mean(iterable)
    values = list(iterable)
    n = len(values)
    return sum(sq(values[i] - m) for i in range(n))


def sdev(iterable: Iterable[float], m=None) -> float:
    """standard deviation of a collection"""
    return sqrt(var(iterable, m=m))

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
