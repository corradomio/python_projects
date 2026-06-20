# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------

def numeric_cmp(a: int|float, b: int|float):
    if a < b: return -1
    if a > b: return +1
    return 0


def string_cmp(a: str, b: str):
    if a < b: return -1
    if a > b: return +1
    return 0


def lexicographic_cmp(s1: str, s2: str) -> int:
    n1 = len(s1)
    n2 = len(s2)
    if n1 < n2: return -1
    if n1 > n2: return +1
    if s1 < s2: return -1
    if s1 > s2: return +1
    return 0

def path_cmp(p1: str, p2: str) -> int:
    s1 = str(p1)
    s2 = str(p2)
    n1 = len(s1)
    n2 = len(s2)
    if n1 < n2: return -1
    if n1 > n2: return +1
    if s1 < s2: return -1
    if s1 > s2: return +1
    return 0


# ---------------------------------------------------------------------------
# Sorters
#
# list.sort(reverse=True|False, key=myFunc)
# sorted(iterable, /, *, key=None, reverse=False
#

def path_sort(iterable):
    return sort_by_comparator(iterable, path_cmp)

def lexicographic_sort(iterable):
    return sort_by_comparator(iterable, lexicographic_cmp)


def sort_by_key(iterable, key, reverse: bool=False):
    return sorted(iterable, key=key, reverse=reverse)


def sort_by_comparator(iterable, comparator):
    import functools
    return sorted(iterable, key=functools.cmp_to_key(comparator))


