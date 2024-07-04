#
# Integer mathematics
#
from typing import Union, Iterable
from math import factorial


IINF: int = 9223372036854775807  # 2^63-1


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

def iseven(n: int) -> bool:
    """e' pari"""
    return n % 2 == 0  # e' pari


def isodd(n: int) -> bool:
    """e' dispari"""
    return n % 2 == 1  # e' dispari


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
    """INVERSE combinations/binomial coefficient"""
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
