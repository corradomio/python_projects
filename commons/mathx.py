from typing import List
from math import *

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
from typing import Union, Iterable, Tuple

INF: float = float('inf')
IINF: int = 9223372036854775807     # 2^63-1


# ---------------------------------------------------------------------------
# chop
# ---------------------------------------------------------------------------

def chop(x: float, l: float, u: float) -> float:
    """Clip x in [l,u]"""
    if x < l: return l
    if x > u: return u
    return x
# end


# ---------------------------------------------------------------------------
# argsort
# ---------------------------------------------------------------------------
# @replaced by itertoolsx::argsort()
# def argsort(values: Iterable, descending: bool = False) -> List[int]:
#     """Sort the values in ascending (ore descending) order and return the indices"""
#     values = list(values)
#     n = len(values)
#     pairs = [(i, values[i]) for i in range(n)]
#     pairs = sorted(pairs, key=lambda p: p[1], reverse=descending)
#     return [p[0] for p in pairs]
# # end


# ---------------------------------------------------------------------------
# to_float
# ---------------------------------------------------------------------------

def to_float(x) -> Union[float, List[float]]:
    """
    Convert, recursively, each object in a float:
    1) int -> float
    2) str -> float
    3) collection -> collection of floats
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, Iterable):
        return list(map(lambda t: to_float(t), x))
    else:
        return float(x)
# end


# ---------------------------------------------------------------------------
# Real comparisons with error
# ---------------------------------------------------------------------------

EPS: float = 1.e-6


def zero(x, eps: float = EPS) -> float:
    """return 0 if the value is smaller than an epsilon"""
    return 0. if -eps <= x <= +eps else x


def sign(x, zero=False, eps: float = EPS) -> int:
    """sign of the number [-1, 0, +1]"""
    if x < -eps: return -1
    if x > +eps: return +1
    return 0 if zero else 1


def isz(x: float, eps: float = EPS) -> bool:
    """is zero"""
    return -eps <= x <= eps


def isnz(x: float, eps: float = EPS) -> bool:
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
    return 0. if isz(x, eps=EPS) else 1. / x


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
    n = len(iterable)
    return sum(sq(iterable[i] - m) for i in range(n))


def sdev(iterable: Iterable[float], m=None) -> float:
    """standard deviation of a collection"""
    return sqrt(var(iterable, m=m))


# ---------------------------------------------------------------------------
# Number Theory
# ---------------------------------------------------------------------------

def ipow(b: Union[int, float], n: int) -> int:
    """b^n with b and n integers"""
    p = 1
    for e in range(n):
        p *= b
    return p


def isqrt(n: int) -> int:
    """integer square root"""
    x0 = 0
    x1 = n
    while abs(x0 - x1) >= 1:
        x0 = x1
        x1 = (x0 + n // x0) // 2
        if x1 > x0:
            return x0
    return x1


def ilog2(n: int) -> int:
    """integer log2"""
    r = -1
    while n != 0:
        r += 1
        n >>= 1
    return r


def isign(x: int, zero: bool = False) -> int:
    """integer sign"""
    if x < 0: return -1
    if x > 0: return +1
    return 1 if zero else 0


# ---------------------------------------------------------------------------
# prime numbers
# ---------------------------------------------------------------------------
from primes import PRIMES


def isprime(n: int) -> bool:
    """check if n is prime"""
    l = isqrt(n)
    i = 0
    while PRIMES[i] <= l:
        if n % PRIMES[i] == 0:
            return False
        else:
            i += 1
    return True


def ifactorize(n: int) -> List[Tuple[int, int]]:
    """factorize the number in [(p, e), ...]
    where p is a prime number and e the exponent > 0"""
    assert n >= 0
    if n < 4:
        return [(n, 1)]
    f = []
    l = isqrt(n)
    i = 0
    while n > l:
        p = PRIMES[i]
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        if e > 0:
            f.append((p, e))
        i += 1
    if n != 1:
        f.append((n, 1))
    return f


def icompose(f: List[Tuple[int, int]]) -> int:
    """compose the integer from the factorization [(p, e), ...]"""
    if len(f) == 0:
        return 0

    n = 1
    for p, e in f:
        n *= ipow(p, e)
    return n


# ---------------------------------------------------------------------------
# product(...)
# ---------------------------------------------------------------------------

def product(iterable: Iterable[Union[float, int]]) -> Union[float, int]:
    """same as sum(...) but for the product"""
    p = 1
    for i in iterable:
        p *= i
    return p


# ---------------------------------------------------------------------------
# integer functions
# ---------------------------------------------------------------------------

def m1pow(n: int) -> int:
    """(-1)^n"""
    return -1 if n & 1 else +1


# ---------------------------------------------------------------------------
# factorials
# ---------------------------------------------------------------------------

# n! = n*(n-1)*...*2*1
def fact(n: int) -> int:
    """n! = n*(n-1)*...*2*1"""
    return factorial(n)


# n!k = n*(n-1)*...(n-k+1)
def failing_fact(n: int, k: int) -> int:
    """failing factorial n!k = n*(n-1)*...(n-k+1)"""
    f = 1
    for i in range(k):
        f *= n
        n -= 1
    return f


# n!k = n*(n+1)*...(n+k-1)
def rising_fact(n: int, k: int) -> int:
    """rising factorial n!k = n*(n+1)*...(n+k-1)"""
    f = 1
    for i in range(k):
        f *= n
        n += 1
    return f


# ---------------------------------------------------------------------------
# Partial Factorial
# ---------------------------------------------------------------------------
# Nota: e' l'inverso del coefficiente binomiale/n di combinazioni
#
#                        n!        (n-0)(n-1)..(n-k+1)
#       comb(n, k) = ---------- = ---------------------
#                     k!(n-k)!     (n-k-0)(n-k-1)..(1)
#
#                      k!(n-k)!         1
#       pfact(n, k) = ---------- = ------------
#                         n!        comb(n, k)
#
#                     k!(n-k-1)!    k!(n-k)!     pfact(n, k)
#       qfact(n, k) = ---------- = ---------- = -------------
#                        n!         n!(n-k)        (n-k)
#
#                      (n-s-t)!s!    (n-s)!s!      n*...(n-(t-1))*(n-t)
#      qfact(n,k,t) = ----------- = --------- * ------------------------
#                      (n-(t-1))!        n!       (n-s)*...((n-s)-(t-1))
#

def comb(n: int, k: int) -> int:
    """combinations/binomial cefficient"""
    if k < 0 or n < k:
        return 0
    if k == 0 or n == k:
        return 1
    c = 1
    while k >= 1:
        c *= n
        c /= k
        n -= 1
        k -= 1
    c = int(round(c, 0))
    return c


def icomb(n: int, k: int) -> float:
    """INVERSE combinations/binomial cefficient"""
    if k < 0 or n < k:
        return 0
    if k == 0 or n == k:
        return 1
    c = 1
    while k >= 1:
        c /= n
        c *= k
        n -= 1
        k -= 1
    return c


def sumcomb(n: int, k: int) -> int:
    """sum(comb(n,i), i=0...k"""
    return sum(comb(n, i) for i in range(k + 1))


def sumcomb1(n: int, k: int) -> int:
    """sum(comb(n-i,1), i=0..k"""
    return sum(comb(n - i, 1) for i in range(k + 1))


# ---------------------------------------------------------------------------
# Quotient Factorial
# ---------------------------------------------------------------------------
# Nota: il termine non e' giusto, perche' qui' abbiamo dei rapporti
#

def qfact(n: int, s: int, t: int = 1) -> float:
    q = comb(n - t, s) * (n - t + 1)
    return 1 / q if q != 0 else 0


# ---------------------------------------------------------------------------
# Stirling numbers
# ---------------------------------------------------------------------------
#
# https://en.wikipedia.org/wiki/Stirling_numbers_of_the_second_kind
#

def stirling2(n: int, k: int) -> int:
    s = 0
    for i in range(k + 1):
        s += m1pow(i) * comb(k, i) * ipow(k - i, n)
    s /= fact(k)
    return int(round(s, 0))


# ---------------------------------------------------------------------------
# Bell numbers
# ---------------------------------------------------------------------------
#
# https://en.wikipedia.org/wiki/Bell_number
#

BELL = [
    1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597,
    27644437, 190899322, 1382958545, 10480142147, 82864869804,
    682076806159, 5832742205057, 51724158235372, 474869816156751
]
BELL_ = dict()


def bell(n: int) -> int:
    b = len(BELL)
    if n < b:
        return BELL[n]
    if n in BELL_:
        return BELL_[n]

    b = 0
    for k in range(n):
        b += comb(n - 1, k) * bell(k)
    BELL_[n] = b
    return b
# end


# ---------------------------------------------------------------------------
# Bernoulli numbers
# ---------------------------------------------------------------------------
#
# https://en.wikipedia.org/wiki/Bernoulli_number
#

BERNULLI = [
    1, -1 / 2, 1 / 6, 0, -1 / 30, 0, 1 / 42, 0, -1 / 30, 0, 5 / 66, 0, -691 / 2730, 0, 7 / 6,
    0, -3617 / 510, 0, 43867 / 798, 0, -174611 / 330
]
BERNULLI_ = dict()


def bernoulli(m: int) -> float:
    if m < len(BERNULLI):
        return BERNULLI[m]
    if m in BERNULLI_:
        return BERNULLI_[m]

    b = 1
    for k in range(m):
        b -= comb(m, k) / (m - k + 1) * bernoulli(k)
    BERNULLI_[m] = b
    return b
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
