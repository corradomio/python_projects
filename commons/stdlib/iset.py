# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
#
# https://graphics.stanford.edu/~seander/bithacks.html
#
from itertools import combinations
from typing import *
import random as rnd

# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------
# def _gen_byte_count_bits():
#     def B2(n): return [n, n + 1, n + 1, n + 2]
#     def B4(n): return B2(n) + B2(n + 1) + B2(n + 1) + B2(n + 2)
#     def B6(n): return B4(n) + B4(n + 1) + B4(n + 1) + B4(n + 2)
#     def B8(n): return B6(n) + B6(n + 1) + B6(n + 1) + B6(n + 2)
#     return B8(0)
#
# _BIT_COUNTS = _gen_byte_count_bits()
#

_BIT_COUNTS = \
    [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3,
     3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
     3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
     4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
     3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6,
     6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5,
     4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8]


def _iset(l: Iterable[int]) -> int:
    """list -> iset"""
    return sum(1 << i for i in l)


def _isetm(m: int, l: Collection) -> int:
    """imask, list -> iset
    :param m: mask
    :param l: elements list
    :return: subset of l as integer
    """
    n = len(l)
    s = 0
    for i in range(n):
        if m & (1 << i):
            s += 1 << l[i]
    return s


def _comb(n: int, k: int) -> int:
    """Combinations/binomial coefficient"""
    if k < 0 or n < k:
        return 0
    if k == 0 or n == k:
        return 1
    c = 1.
    while k >= 1:
        c *= n
        c /= k
        n -= 1
        k -= 1
    c = int(round(c, 0))
    return c


comb = _comb


# ---------------------------------------------------------------------------
# Low level routines
# ---------------------------------------------------------------------------

def ipow(b: Union[int, float], n: int) -> int:
    """Integer power: b^n with b and n integers"""
    p = 1
    for e in range(n):
        p *= b
    return p


def ilog2(p: int) -> int:
    """Integer logarithm based 2"""
    l = -1
    while p != 0:
        l += 1
        p >>= 1
    return l


def ihighbit(S: int) -> int:
    """
    Index of the highest bit
    For 0, return -1

    :param S: bit set
    :return: bit position or -1
    """
    h = -1
    while S != 0:
        h += 1
        S >>= 1
    return h


def ilowbit(S: int) -> int:
    """
    Index of the lowest bit
    For 0, return -1

    :param S: bit set
    :return: bit position or -1
    """
    if S == 0:
        return -1
    b = 0
    while S and S & 1 == 0:
        b += 1
        S >>= 1
    return b


# ---------------------------------------------------------------------------
# Function bases
# ---------------------------------------------------------------------------

def idsign(A: int, B: int = 0) -> int:
    """(-1)^|A - B|    (difference sign)"""
    D = idiff(A, B)
    c = icard(D)
    return -1 if c & 1 else +1


def iisign(A: int, B: int) -> int:
    """(-1)^|A * B|    (intersection sign)"""
    I = iinterset(A, B)
    c = icard(I)
    return -1 if c & 1 else +1


def iusign(A: int, B: int) -> int:
    """(-1)^|A + B|    (union sign)"""
    I = iunion(A, B)
    c = icard(I)
    return -1 if c & 1 else +1


def isdsign(A: int, B: int) -> int:
    """(-1)^|A <> B|    (symmetric difference sign)"""
    I = isdiff(A, B)
    c = icard(I)
    return -1 if c & 1 else +1


# -----------------------------------------------------------------------------
# Syntax:   b_S(T)
#

# unanimity game: 1 if T >= S else 0
def ug(S, T): return int(iissubset(S, T))


# dirac: 1 if T == S else 0
def dirac(S, T): return int(S == T)


# walsh: (-1)^|S-T|
def walsh(S, T): return idsign(S, T)


# ---------------------------------------------------------------------------
# Lexicographic order
# ---------------------------------------------------------------------------

def ilexcount(k: int, n: int) -> int:
    """number of sets with cardinality in the range [0,k]"""
    if k < 0: k = n
    return sum(_comb(n, i) for i in range(k+1))


def ilexorder(l: int, n: int) -> int:
    """maximum cardinality used in the list of l lexicographic elements"""
    k = -1
    c = 0
    while c < l:
        k += 1
        c += _comb(n, k)
    return k


def ilexset(L: int, n: int) -> int:
    """lexicographic index to set"""
    S = 0
    k = -1
    nk = _comb(n, k)
    while nk <= L:
        L -= nk
        k += 1
        nk = _comb(n, k)
        continue

    while k > 0:
        ck = 0
        ckk = _comb(ck, k)
        while ckk <= L:
            ck += 1
            ckk = _comb(ck, k)
        ck -= 1
        L -= _comb(ck, k)
        S = iadd(S, ck)
        k -= 1
    return S


def ilexidx(S: int, n: int) -> int:
    """set to lexicographic index"""
    m = icard(S)
    L = 0
    for k in range(m):
        L += _comb(n, k)
    i = 0
    # for e in imembers(S):
    #     i += 1
    #     ci = _comb(e, i)
    #     l += ci
    for e in range(n):
        if S & 1:
            i += 1
            ci = _comb(e, i)
            L += ci
        elif S == 0:
            break
        S >>= 1
    return L


# ---------------------------------------------------------------------------

def ilexpowersetn(n: int, empty: bool = True, full: bool = True,
                  k: Optional[Union[tuple, list[int], int]] = None) -> Iterator[int]:
    N = (1 << n) - 1
    return ilexpowerset(N, empty=empty, full=full, k=k)


def ilexpowerset(N: int,
                 empty: bool = True,
                 full: bool = True,
                 k: Optional[Union[tuple, list[int], int]] = None) -> Iterator[int]:
    """
    subsets in lexicographic order with cardinality in the range specified by k
    :param N: full set
    :param empty: if to include the empty set
    :param full: if to include the full set
    :param k: card | (min_card,max_card)
    :return:
    """
    n = icard(N)
    if k is None:
        k = 0 if empty else 1, n if full else n-1
    return ilexsubset(0, N, n=n, k=k)


ipowerset_lex = ilexpowerset


def ilexsubset(B: Optional[int] = None,
               E: Optional[int] = None,
               n: Optional[int] = None,
               k: Optional[Union[tuple, list[int], int]] = None) -> Iterator[int]:
    """
    Subsets in the range [B,E] with cardinality in the range [kmin, kmax]
    in lexicographic order
    :param int B: begin set
    :param int E: end set
    :param int n: n of elements in the full set (N=[0,..n-1])
    :param k: cardinality
        None: (0, n)
        int:  (k, k)
        [int]:(0, k)
        (kmin,kmax)
    :return: iterator
    """
    from itertools import combinations

    # assert E is not None or n is not None
    if B is None and E is None and n is None:
        raise ValueError("Missing B, E, n")
    if B is None and E is None:
        B, E = 0, (1 << n) - 1
    if E is None:
        B, E = 0, B
    if n is None:
        n = icard(E)
    b = icard(B)
    kmin, kmax = parse_k(k, n, b)
    D = idiff(E, B)
    L = ilist(D)
    for k in range(kmin-b, kmax-b+1):
        for C in combinations(L, k):
            S = _iset(C)
            U = iunion(B, S)
            yield U


isubsets_lex = ilexsubset


# ---------------------------------------------------------------------------

def parse_k(k: Union[None, int, list[int], tuple], n: int, b: int = 0) -> tuple[int, int]:
    """
    Parse k
        None        -> [0, n]
        int         -> [k, k]
        [int]       -> [0, k]
        [int,int]   - [kmin, kmax]

    :param k: what to parse
    :param n: set's cardinality
    :param b: initial value of the range
    """
    if k is None:
        kmin, kmax = 0, n
    elif isinstance(k, int):
        kmin, kmax = k, k
    elif len(k) == 1:
        kmin, kmax = 0, k[0]
    else:
        kmin, kmax = k
    if kmax < 0:
        kmax = n + kmax
    return max(b, kmin), kmax


# ---------------------------------------------------------------------------
# Integer sets
# ---------------------------------------------------------------------------
# Each set is defined by the bits in a integer
# This means that the elements of the set are the numbers 0...(2^MAX_BITS)-1
# It is not available the operation icompl(s) of a bitset because it is not
# available the information about the TOTAL the number of elements in the
# bitset but is is available as icompl(c | X) = idiff(X, s)
#

def ibinset(bits: Iterable[int]) -> int:
    """
    Binary set (composed by {0,1}) to integer

        [0,1,1,0] -> 6:int
         0 1 2 3
         1 2 4 8
    """
    S = 0
    m = 1
    for i in bits:
        if i:
            S |= m
        m <<= 1
    return S


def ibinlist(S: int, n: int) -> list[int]:
    """
    Convert ibitset S i a list of bits:
    :param S:
    :param n:
    :return:
    """
    M = (1 << n)
    L = [0]*n
    for i in ilist(S):
        L[i] = 1
    return L


def isetn(n: int) -> int:
    """Full set"""
    return (1 << n) - 1


def iset(L: Iterable[int]) -> int:
    """
    Convert the list into a integer

        (1,2) -> 6

    :param L: list of elements
    :return: set as integer
    """
    assert type(L) in [list, set, tuple, range]
    return _iset(L)


def ilistset(L: Iterable[int]) -> int:
    """
    Convert the list of bits into a integer

        [1,1,0,0] -> 3

    :param L: list of flags
    :return: set as integer
    """
    assert type(L) in [list, tuple]
    S = 0
    F = 1
    for l in L:
        if l:
            S += F
        F *= 2
    return S
# end


def ilist(S: int) -> list[int]:
    """
    Convert a bitset in the tuple of elements

        6 -> (1,2)

    :param S: bitset
    :return: list of elements
    """
    L = list(imembers(S))
    return L


def ilist_tuple(S: int) -> list[int]:
    return list(imembers(S))


def ilist_set(S: int) -> Set[int]:
    return set(imembers(S))


def ilist_list(S: int) -> List[int]:
    return list(imembers(S))


# ---------------------------------------------------------------------------
# Element operations
# ---------------------------------------------------------------------------

def icard(S: int) -> int:
    """Cardinality of the bitset: n of elements in the set"""
    c = 0
    while S != 0:
        c += _BIT_COUNTS[S & 0xFF]
        S >>= 8
    return c


icount = icard


def iinsert(S: int, i: int) -> int:
    """Add the element i into the bitset"""
    return S | (1 << i)


iadd = iinsert


def iremove(S: int, i: int) -> int:
    """Remove the element i from the bitset"""
    a = 1 << i
    if S & a:
        S = S ^ a
    return S


isub = iremove


def ireplace(S: int, i: int, j: int) -> int:
    """Remove the element 'i' and add the element 'j'"""
    return iinsert(iremove(S, i), j)


# ---------------------------------------------------------------------------
# Special operations
# ---------------------------------------------------------------------------

def isplit(S: int) -> tuple[int, int]:
    """Split the set in two parts"""
    s = icard(S)
    t = s//2
    T = 0
    c = 0
    for e in imembers(S):
        if c == t:
            break
        T = iadd(T, e)
        c += 1
    # end
    return T, idiff(S, T)


def ireduceset(S: int, P: int, n: int, L=None):
    """
    Reduce the set S removing the elements in P AND rearranging the indices
    Note: the elements in P are COLLAPSED into the element with the minimal index 'c'

    :param S: set
    :param P: reduction set (as bitset)
    :param n: n of elements in the set
    :param L: P in form of list
    :return: reduced set
    """
    assert icard(P) > 1
    N = (1 << n) - 1

    exists = ihasintersect(S, P)
    c = ilowbit(P)
    # c: COLLAPSED element for elements in P

    if L is None:
        L = ilist(P)[1:]
    l = len(L)

    R = S
    for i in range(l):
        M = ((1 << L[i]) - 1) >> i
        F = N ^ (M << 1 | 1)
        R = (R & M) | ((R & F) >> 1)

    if exists:
        R = iadd(R, c)

    return R


# ---------------------------------------------------------------------------
# Set predicates
# ---------------------------------------------------------------------------

def iismember(S: int, i: int) -> bool:
    """Check if the element i is in the bitset"""
    return S & (1 << i) != 0


def iissameset(S1: int, S2: int) -> bool:
    """Check if the two sets are equal"""
    return S1 == S2


def iissubset(S1: int, S2: int) -> bool:
    """Check is S1 is a subset of S2"""
    return S1 & ~S2 == 0


def iissuperset(S1: int, S2: int) -> bool:
    """Check if S1 is a superset of S2"""
    return S2 & ~S1 == 0


def ihasintersect(S1: int, S2: int) -> bool:
    """Check if S1 has intersection with S2"""
    return S1 & S2 != 0


# ---------------------------------------------------------------------------
# Set Operations
# ---------------------------------------------------------------------------

def imember(S: int, i: int) -> int:
    """1 if i is in member of S else 0"""
    return 1 if S & (1 << i) else 0


def iunion(S1: int, S2: int) -> int:
    """union"""
    return S1 | S2


def iinterset(S1: int, S2: int) -> int:
    """intersection"""
    return S1 & S2


def idiff(S1: int, S2: int) -> int:
    """difference"""
    return S1 & ~S2


def isdiff(S1: int, S2: int) -> int:
    """symmetric difference"""
    return (S1 & ~S2) | (~S1 & S2)


def idiffn(S1: int, S2: int, S3: int=0) -> int:
    """difference between 3 bitsets (S1 - S2) - S3"""
    return S1 & ~S2 & ~S3


def idiff_gt(S1: int, S2: int) -> int:
    l = ilowbit(S2)+1
    return S1 & ~S2 & (-1 << l)


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------

def imembers(S: int, reverse: bool=False) -> Iterable[int]:
    """
    Iterator on the set's members

    :param S: bitset
    :param reverse: if to scan the bitset in the reverse order
    :return: an iterator on the elements
    """
    if S == 0:
        return
    else:
        h = ihighbit(S)

    if not reverse:
        i = 0
        while i <= h:
            if S & (1 << i):
                yield i
            i += 1
    else:
        i = h
        while i >= 0:
            if S & (1 << i):
                yield i
            i -= 1
    return


# ---------------------------------------------------------------------------
# Subsets & powersets
# ---------------------------------------------------------------------------

def ipowerset(N: int, empty: bool = True, full: bool = True) -> Iterator[int]:
    """all subsets of the fullset N (in binary order)"""
    s = 0 if empty else 1
    e = N if full else (N - 1)
    return range(s, e + 1)


def ipowersetn(n: int, empty: bool = True, full: bool = True) -> Iterator[int]:
    N = (1 << n) - 1
    return ipowerset(N, empty=empty, full=full)


def isubsets(B: int, E: Optional[int] = None, lower: bool = True, upper: bool = True):
    """
    Subsets in the range [B, E].
    If card(B) > card(E), generate the subsets in the range [E, B]
    :param B: begin set
    :param E: end set
    :param lower: if to include the lower set
    :param upper: if to include the upper set
    :return: iterator on S, starting from B, with B <= S <= E
    """
    if E is None:
        B, E = 0, B
    D = ilist(idiff(E, B))
    d = 1 << len(D)

    if icard(B) <= icard(E):
        b, e, s = 0, d, 1
        if not lower: b += 1
        if not upper: e -= 1
    else:
        b, e, s = d-1, -1, -1
        if not lower: e += 1
        if not upper: b -= 1
    for S in range(b, e, s):
        U = _isetm(S, D)
        yield iunion(B, U)


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

def _algorithm_u(ns: Collection[int], m: int) -> Iterator[int]:

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    if m <= 1:
        return [[ns]]

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)


def ipartitions(S: int, size=None, reverse=False) -> Iterator[int]:
    """
    Generate the partitions of the bitset
    If reverse is False, generate the partitions from 1 to n elements
    otherwise in reverse order

    :param S: bitset
    :param size: number of elements in partition
                    None: from 1 to n
                    int:  only size elements
                    (min,max): from min to max elements
    :param reverse: it to generate the partitions in the
                    reverse order
    :return: partitions generator
    """
    ls = ilist(S)   # convert the set in the list of elements
    ns = len(ls)    # number of elements in the list

    # b: begin
    # e: end
    # s: step

    if isinstance(size, int):
        b, e, S = size, size + 1, 1
    elif type(size) in [list, tuple]:
        b, e = size
        S = 1
        assert b < e
        e += 1
    else:
        b, e, S = 0, ns + 1, 1
    if reverse:
        b, e, S = e - 1, b - 1, -1

    for m in range(b, e, S):
        for P in _algorithm_u(ls, m):
            yield tuple(map(_iset, P))


# ---------------------------------------------------------------------------
# Special iterators
# ---------------------------------------------------------------------------

def idisjointpairs(N: int, empty=True) -> Iterator[tuple[int, int]]:
    """
    Generate all possible pairs of disjoint sets
    :param N: fullset
    :param empty: if to include the empty sets
    :return: pair generator
    """
    n = icard(N)
    b = 0 if empty else 1
    for S in ilexsubset(N, k=[b, n//2]):
        s = icard(S)
        D = idiff_gt(N, S)
        for T in ilexsubset(D, k=[s, n]):
            yield S, T


def ipowersetpairs(N: int, empty=True) -> Iterator[tuple[int, int]]:
    """
    Generate all possible pairs of sets with some intersection
    :param N: fullset
    :param empty: if to include the empty sets
    :return: pair generator
    """
    # for S in ipowerset(N, empty=empty):
    #     for T in ipowerset(N, empty=empty):
    #         yield S, T
    # for S, T in idisjointpairs(N, empty=empty):
    #     U = iunion(S, T)
    #     for R in isubsets(U):
    #         SR = iunion(S, R)
    #         TR = iunion(T, R)
    #         if ilowbit(SR) > ilowbit(TR):
    #             continue
    #         yield SR, TR
    for S, T in idisjointpairs(N, empty=empty):
        D = idiffn(N, S, T)
        for I in ilexsubset(D):
            yield iunion(S, I), iunion(T, I)


def isupersetpairs(N: int, same=True, empty=True) -> Iterator[tuple[int, int]]:
    """
    Generate all possible pairs of sets and supersets of the first set
    :param N: fullset
    :param same: if to include the same set in the list of supersets
    :return: pair generator
    """
    b = 0 if empty else 1
    for S in ipowerset(N, empty=empty):
        for T in isubsets(S, N, lower=same):
            yield S, T
    # n = icard(N)
    # for S in ilexsubset(N, k=[0 if empty else 1, n//2]):
    #     for T in isubsets(S, N, lower=same):
    #         yield S, T
# end


def isubsetpairs(N: int, same=True, empty=True) -> Iterator[tuple[int, int]]:
    """
    Generate all possible pairs of sets and subsets of the first set
    :param N: fullset
    :param same: if to include the same set in the list of subsets
    :return: pair generator
    """
    for S in ipowerset(N, empty=empty):
        for T in isubsets(S, lower=empty, upper=same):
            yield T, S


# ---------------------------------------------------------------------------
# Random sets
# ---------------------------------------------------------------------------

def irandset(n: int, rnd=None) -> int:
    """
    Generate a random set with maximum cardinality n

    :param n: cardinality
    :param rnd: random number generator
    :return: random set as integer
    """
    from random import Random
    if rnd is None:
        rnd = Random()
    if not isinstance(rnd, Random):
        rnd = Random(rnd)

    S = 0
    for i in range(n):
        r = rnd.random()
        if r < .5:
            S = iadd(S, i)
    return S


# def irandsubsets(n: int, n_perm: int) -> list:
#     """
#     Generate a list of subsets using random permutationz=s
#     :param n: number of elements in the set
#     :param n_perm:  number of permutations
#     :return:
#     """
#     from random import shuffle
#
#     members = list(range(n))
#     subsets = set()
#
#     for i in range(n_perm):
#         for j in range(n + 1):
#             S = _iset(members[0:j])
#             subsets.add(S)
#         shuffle(members)
#     # end
#     return list(subsets)
# # end


# ---------------------------------------------------------------------------
# Support
# ---------------------------------------------------------------------------

def ijaccard_index(S1: int, S2: int) -> float:
    """
    Jaccard index

        cardinality(intersection(S1,S2))
        -------------------------------
        cardinality(union(S1,S2))

    :param S1: first bitset
    :param S2: second bitset
    :return:
    """
    if S1 == 0 and S2 == 0:
        return 1.
    else:
        return icard(iinterset(S1, S2)) / icard(iunion(S1, S2))


def ihamming_distance(S1: int, S2: int) -> int:
    """
    Hamming distance

        cardinality(simmetric_difference(S1, S2))

    :param S1: first bitset
    :param S2: second bitset
    :return:
    """
    return icard(isdiff(S1, S2))


# ---------------------------------------------------------------------------
# Extras/compatibility
# ---------------------------------------------------------------------------

def ipowersett(n: int, empty=True, full=True, df=False) -> Iterator[int]:
    """
    Generate all subsets of the set {0,1,...n-1} using the traversal:

        - depth first    (df=True)
        - breadth first  (df=False)

    order

    :param n: n of elements in the set
    :param empty: if include the empty set
    :param full:  if include the full set
    :param df: depth first/breadth first
    :return: an iterator
    """
    p = 1 << n
    N = p - 1
    b = [0]*p

    if empty:
        stk = [0]
    else:
        b[0] = 1
        stk = [(1 << i) for i in range(n)]
    if not full:
        b[-1] = 1

    while stk:
        S = stk[0]
        stk = stk[1:]
        yield S

        D = idiff(N, S)
        if df:
            # depth first
            for e in imembers(D, reverse=True):
                T = iadd(S, e)
                if not b[T]:
                    stk = [T] + stk
                    b[T] = 1
        else:
            # breadth first
            for e in imembers(D):
                T = iadd(S, e)
                if not b[T]:
                    stk = stk + [T]
                    b[T] = 1
        # end
    # end
# end


def isubsetsc(S: int, l=None, u=None) -> Iterator[int]:
    """
    Generate the subsets of S with a number of elements in the range [l,u]

    :param S: bitset
    :param l: lower number of elements
    :param u: upper number of elements
    :return: an iterator
    """
    S = ilist(S)
    n = len(S)
    if l is None and u is None:
        l, u = 0, n
    if u is None:
        u = l

    for k in range(l, u+1):
        for c in combinations(S, k):
            yield iset(c)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def imapset(M : Union[List[List[int]], Dict[int, List[int]]]) -> List[int]:
    """
    Convert a map

        e -> [a1,...]

    in a list where e is used as index and [a1,...] is converted in a ibitset

    :param M: map
    :return: list of ints
    """
    if type(M) in [dict]:
        n = max(M.keys())+1
        T = [None]*(n)
        for k in M.keys():
            T[k] = M[k]
        M = T
    n = len(M)
    S = [0]*(n)
    for i in range(n):
        S[i] = iset(M[i])
    # end
    return S


# ---------------------------------------------------------------------------
# Boolean functions
# ---------------------------------------------------------------------------
# There are 2^2^n binary functions with n parameters
# More in general:  k^k^n k-functions with n parameters
#

# def iboolfun(k: int, n: int) -> list[int]:
#     """
#     Return the k-th boolean function in n variables
#     If k is -1, it generates a random function
#
#     :param k: k-th boolean function, or -1
#     :param n: n of parameters
#     :return: list of {0,1}
#     """
#     M = (1 << n)
#     if k == -1:
#         T = [1 if rnd.random() >= 0.5 else 0 for _ in range(M)]
#     else:
#         T = ibinlist(k, n)
#     return T


def ibooltable(k: int, n: int) -> list[int]:
    """
    Return the k-th boolean table in n variables
    If k is -1, it generates a random function

    :param k: k-th boolean function, or -1
    :param n: n of parameters
    :return: list of {0,1}
    """
    if k == -1:
        L = 2**(2**n)
        k = rnd.randrange(0, L)
    # end
    N = (1 << n) - 1
    T = [0]*(N + 1)
    for S in isubsets(N):
        T[S] = imember(k, S & N)
    return T
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
