from typing import Iterable


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def list_map(f, l):
    """
    Same as 'list(map(f, l))'
    """
    return list(map(f, l))


# ---------------------------------------------------------------------------
# lrange
# ---------------------------------------------------------------------------

def lrange(start, stop=None, step=1) -> list[int]:
    """As 'range' but it returns a list"""
    if stop is None:
        return list(range(start))
    else:
        return list(range(start, stop, step))
# end


# ---------------------------------------------------------------------------
# argsort
# ---------------------------------------------------------------------------

def argsort(values: Iterable, descending: bool = False) -> list[int]:
    """Sort the values in ascending (ore descending) order and return the indices"""
    n = len(list(values))
    pairs = [(i, values[i]) for i in range(n)]
    pairs = sorted(pairs, key=lambda p: p[1], reverse=descending)
    return [p[0] for p in pairs]
# end


# ---------------------------------------------------------------------------
# Numerical aggregation functions
# ---------------------------------------------------------------------------
# sum_  as Python 'sum([...])'
# mul_  as Python 'sum([...]) but for multiplication
#

def sum_(x):
    """
    A little more flexible variant of 'sum', supporting None and numerical values
    """
    if x is None:
        return 0
    elif isinstance(x, (int, float)):
        return x
    else:
        return sum(x)


def prod_(x):
    """
    Multiplicative version of 'sum' supporting None and numerical values
    """
    if x is None:
        return 1
    elif isinstance(x, (int, float)):
        return x
    else:
        m = 1
        for e in x:
            m *= e
        return m

# compatibility
mul_ = prod_
