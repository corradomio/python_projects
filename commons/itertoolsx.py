#
# List, tuple, permutation extensions
#
# product
# permutations
# combinations.
from itertools import combinations
from typing import Collection, Iterator


# ---------------------------------------------------------------------------
# argsort
# ---------------------------------------------------------------------------

def argsort(l: list, reverse: bool = False) -> list:
    """
    Order the list and return the list of indices ordered by data values

    :param l: data values
    :param key: function to extract the value from the  list's element
    :param reverse: if ordering in reverse order
    :return: list of ordered indices
    """

    n = len(l)
    ipairs = [(i, l[i]) for i in range(n)]
    opairs = sorted(ipairs, key=lambda p: p[1], reverse=reverse)
    output = [p[0] for p in opairs]
    return output
# end


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------
def list_map(f, l: list) -> list:
    """
    Apply the function to all list's elements

    :param lambda f: function to apply to all elements of the list
    :param list l: list of items to transform
    :return list: transformed list
    """
    return list(map(f, l))
# end


def dict_map(f, d: dict, kf=None) -> dict:
    """
    Apply the function to all dict's values.
    It is possible to apply a function also to dict's keys

    :param lambda f: function to apply to the dictionary's values
    :param lambda kf: function to apply to the dictionary's keys
    :param dict d: dictionary to transform
    :return: transformed dictionary
    """
    if kf is not None:
        return {kf(k): f(v) for k, v in d.items()}
    else:
        return {k: f(v) for k, v in d.items()}
# end


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------
def dict_to_list(d: dict) -> list:
    """
    Convert a dictionary in a list of pairs/tuples

    :param d: dictionary
    :return: list of tuples
    """
    l = []
    for k in d:
        l.append( (k, d[k]) )
    return l
# end


# ---------------------------------------------------------------------------
# List handling
# ---------------------------------------------------------------------------
def list_len(l):
    return 0 if l is None else len(l)
# end


def sublists(l):
    return (l[0:i] for i in range(len(l)+1))
# end


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------
#   issubset(S1,S2)
#   isempty(S)
#   powerset(N, empty=True, full=True) -> Iterator
#   subsets(N, k=None) -> Iterator
#   set_range(B, E, k) -> Iterator
#
#   powersetn(n, empty=True, full=True)
#   subsetsn(n, k=None)
#


def issubset(S1: Collection, S2: Collection) -> bool:
    """
    Check if S1 is a subset of S2

    :param list s1: first set
    :param list s2: second set
    :return: True if l1 is a subset of l2
    """
    for s in S1:
        if s not in S2:
            return False
    return True
# end


def isempty(S: Collection) -> bool:
    """
    Check if s is None or the empty set/list

    :param S: list or tuple
    :return: if is empty
    """
    return S is None or len(S) == 0
# end


def powerset(N: Collection, empty=True, full=True) -> Iterator:
    # N = list(N)
    # n = len(N)
    # p = 1 << n
    # b = 0 if empty else 1
    # e = p if full else p - 1
    # for s in range(b, e):
    #     yield tuple(N[i] for i in range(n) if s&(1 << i))
    k = 0 if empty else 1, 0 if full else -1
    return subsets(N, k=k)
# end


def subsets(B, E=None, k=None) -> Iterator:
    """
    :param N:
    :param k:
        None -> (0,n)
        k -> (k,k)
        [k] -> (0,k)
        [kmin,kmax]
    :return:
    """
    def _parsek(k, n, b=0):
        if k is None:
            kmin,kmax = 0, n
        elif isinstance(k, int):
            kmin, kmax = k, k
        elif len(k) == 1:
           kmin, kmax = 0, k[0]
        else:
            kmin, kmax = k
        if kmax <= 0:
            kmax += n
        return max(b, kmin), kmax
    # end

    if E is None:
        B, E = [], B

    if len(E) < len(B):
        B, E = E, B
        B = intersect(B, E)
        D = difference(E, B)
        n = len(E)
        b = len(B)

        kmin, kmax = _parsek(k, n, b)
        for k in range(kmin-b, kmax-b+1):
            for C in combinations(D, k):
                yield difference(E, C)
    else:
        n = len(E)
        D = difference(E, B)
        b = len(B)

        kmin, kmax = _parsek(k, n, b)
        for k in range(kmin-b,kmax-b+1):
            for C in combinations(D, k):
                yield union(B, C)
# end


# def powersetn(n, empty=True, full=True):
#     kmin = 0 if empty else 1
#     kmax = n if full else -1
#     return subsets(range(n), k=(kmin, kmax))
# # end


# def subsetsn(n, k=None):
#     return subsets(range(n), k=k)
# # end


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------
# convert the set N=[0,1...n-1] to an integer
#
#   bit_set/bit_index
#   bin_set/bin_index       binary_set/binary_index
#   lex_set/lex_index       lexicographic_set/lexicographic_index
#

def binary_index(S: Collection, n: int) -> int:
    """[e1,e2,...] -> m, ei in {0,1,...n-1} """
    m = 0
    for i in S:
        assert i < n
        m |= 1 << i
    return m
# end


def binary_set(m: int, n) -> Collection:
    """m -> [e1,e2,...]"""
    S = []
    for i in range(n):
        if m & 1:
            S.append(i)
        m >>= 1
    return tuple(S)
# end


bin_index = binary_index
bin_set = binary_set


# ---------------------------------------------------------------------------
# _combinatorial Number System
#   - _combinadics
#   - Gosper's hack (item 175)
#   - Banker's sequence
#
#   https://en.wikipedia.org/wiki/_combinatorial_number_system
#   https://www.developertyrone.com/blog/generating-the-mth-lexicographical-element-of-a-mathematical-_combination/
#   .
# ---------------------------------------------------------------------------
#
#   []      0
#   [0]     1
#   [1]     2
#   [2]     3
#   [0,1]   4
#   [1,2]   5
#   [0,2]   6
#   [0,1,2] 7.
#
#

def _comb(n: int, k: int) -> int:
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
# end


def lexicographic_index(c: Collection, n: int) -> int:
    """
    Convert a collection of integers (in range [0..n-1] into the lexicographic index
    :param Collection c: a collection of integers
    :param int n: total number of elements
    :return: lexicographic index
    """
    m = len(c)
    l = 0
    for k in range(m):
        l += _comb(n, k)
    for i in range(m):
        ci = _comb(c[i], (i+1))
        l += ci
    return l
# end


def lexicographic_set(l: int, n: int) -> Collection:
    """
    Convert the integer l into the collection in the specified position
    :param int l: lexicographic index
    :param int n: total number of elements
    :return: collection
    """
    s = []

    k = -1
    nk = _comb(n, k)
    while nk <= l:
        l -= nk
        k += 1
        nk = _comb(n, k)

    while k > 0:
        ck = 0
        ckk = _comb(ck, k)
        while ckk <= l:
            ck += 1
            ckk = _comb(ck, k)
        ck -= 1
        l -= _comb(ck, k)
        s.append(ck)
        k -= 1
    s.reverse()
    return s
# end


def lexicographic_range(k: int, n: int) -> tuple:
    """
    Indices range for collections with k elements choose from n
    :param int k: number of elements in the collection
    :param int n: total number of elements
    :return:
    """
    at = sum(_comb(n, i) for i in range(k))
    return at, at+_comb(n, k)
# end


lex_set = lexicographic_set
lex_index = lexicographic_index


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

def sorted_tuple(l):
    return tuple(sorted(l))
# end


def intersect(s1, s2):
    return tuple(set(s1).intersection(s2))
# end


def difference(s1, s2):
    return tuple(set(s1).difference(s2))
# end


def union(s1, s2):
    return tuple(set(s1).union(s2))
# end


def simdiff(s1, s2):
    return union(difference(s1, s2), difference(s2, s1))
# end


def replace_with(S, i, R, j):
    return tuple(S[0:i] + R[j:j+1] + S[i+1:])
# end


# ---------------------------------------------------------------------------
# Set distances
# ---------------------------------------------------------------------------

def jaccard_index(s1, s2):
    if isempty(s1) and isempty(s2):
        return .0
    else:
        return len(intersect(s1, s2))/len(union(s1, s2))
# end


def jaccard_distance(s1, s2):
    return 1.0 - jaccard_index(s1, s2)
# end


def hamming_distance(s1, s2):
    if isempty(s1) and isempty(s2):
        return .0
    else:
        return len(simdiff(s1, s2))/len(union(s1, s2))
# end


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------

#
# Partitions
#

def algorithm_u(ns, m):

    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        # return ps
        return list(map(tuple, ps))

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
        return [[tuple(ns)]]

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)
# end


def partitions(it, k=None, reverse=False):
    """
    Generate all partitions of the specified collection.
    It is possible to specify the minimum and maximum number of elements that
    each partition must contain.

    if minl and maxl are None:     minl = 0, maxl=len(collection)
    if maxl is None: maxl = minl

    To ensure consistency, the iterable is converted in a SORTED LIST
    ad each set is a tuple

    Examples:

        > for p in partitions([2,1])


    :param it: collection in for of iterable object
    :param k:
        None -> (0,n)
        k -> (k,k)
        [k] -> (0,k)
        [kmin,kmax]
    :param reverse: is to generate the partitions in reverse order
    :return: a iterator on the partitions
    """
    N = sorted(it)
    n = len(N)

    def _parsek(k, n, b=0):
        if k is None:
            kmin,kmax = 0, n
        elif isinstance(k, int):
            kmin, kmax = k, k
        elif len(k) == 1:
           kmin, kmax = 0, k[0]
        else:
            kmin, kmax = k
        if kmax <= 0:
            kmax += n
        return max(b, kmin), kmax
    # end

    kmin, kmax = _parsek(k, n)

    if not reverse:
        begin, end, step = kmax, kmin - 1, -1
    else:
        begin, end, step = kmin, kmax+1, +1

    for m in range(begin, end, step):
        for p in algorithm_u(N, m):
            yield p
# end


# ---------------------------------------------------------------------------
# Subsets
# ---------------------------------------------------------------------------

# def powersetn(n, empty=True, full=True):
#     """
#     Generate the subsets of a set with 'n' elements
#
#     :param n: number of elements in the set
#     :param empty: if to include the empty set
#     :param full:  if to include the full set
#     :return: a generator on the subsets
#     """
#     p = 1 << n
#     s = 0 if empty else (0 + 1)
#     e = p if full  else (p - 1)
#
#     N = list(range(n))
#
#     bpos = [1 << i for i in range(n)]
#
#     for i in range(s, e):
#         yield tuple(N[j] for j in range(n) if (i & bpos[j]))
# # end


# def subsetsn(n, minl=None, maxl=None):
#     return subsetsc(range(n), minl=minl, maxl=maxl)


# def subsets(it1, it2=None, empty=True, full=True):
#     """
#     Return an iterator on all subsets between it1 and it2
#
#     If it2 is None, the range is from the empty set to it1
#
#     If empty is false, skip the empty set
#     If full  is false, skip the fullset
#
#     :param it1: source set
#     :param it2: destination set
#     :param empty: if to include the empty set
#     :param full:  if to include the full set
#     :return: a generator on the subsets
#     """
#     if it2 is None:
#         it1, it2 = [], it1
#
#     A = set(it1)
#     S = list(set(it2).difference(A))
#     p = len(S)
#     n = 1 << p
#     b = 0 if empty else 1
#     e = n if full else n - 1
#
#     bpos = [1 << i for i in range(p)]
#
#     if len(A) == 0:
#         for i in range(b, e):
#             yield tuple(S[j] for j in range(p) if (i & bpos[j]))
#     else:
#         for i in range(b, e):
#             yield A.union(S[j] for j in range(p) if (i & bpos[j]))
# # end


# def subsetsc(it, minl=None, maxl=None):
#     """
#     Generate all subsets of the specified collection.
#     It is possible to specify the minimum and maximum number of elements that
#     each subset must contain.
#
#     if minl and maxl are None:     minl = 0, maxl=len(collection)
#     if maxl is None: maxl = minl
#
#     To ensure consistency, the iterable is converted in a SORTED LIST
#     ad each set is a tuple
#
#     Examples:
#
#         > for s in subsets([2,1])
#
#
#     :param it: collection in for of iterable object
#     :param minl: minimum size of the subsets
#     :param maxl: maximu size of the subsets
#     :param reverse: is to generate the subsets in reverse order
#     :return: a iterator on the subsets
#     """
#     from itertools import combinations
#
#     ns = sorted(list(it))
#     n = len(it)
#     if minl is None and maxl is None:
#         minl, maxl = 0, n
#     if maxl is None:
#         maxl = minl
#     if maxl and maxl < 0:
#         maxl = n + maxl
#     if maxl is None or maxl < minl:
#         maxl = minl
#     if maxl > n:
#         maxl = n
#     begin, end, step = minl, maxl+1, +1
#
#     for r in range(begin, end, step):
#         for ss in combinations(ns, r):
#             yield ss
#     # raise StopIteration()
# # end
