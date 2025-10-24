from multimethod import multimethod, overload
import gurobipy as grb
import numpy as np


# ---------------------------------------------------------------------------

@multimethod
def rank(a: np.ndarray) -> int:
    return len(a.shape)


@multimethod
def rank(l: list) -> int:
    if isinstance(l, list) and len(l) > 0 and isinstance(l[0], list):
        return 2
    if isinstance(l, list):
        return 1
    else:
        return 0

@multimethod
def rank(tdict: grb.tupledict) -> int:
    return len(list(tdict.keys())[0])



# ---------------------------------------------------------------------------

@multimethod
def shape(a: np.ndarray) -> tuple[int, ...]:
    return a.shape


@multimethod
def shape(tdict: grb.tupledict) -> tuple[int, ...]:
    last_index: list[int] = sorted(tdict.keys())[-1]
    return tuple(c+1 for c in last_index)


@multimethod
def shape(l: list) -> tuple[int, ...]:
    if isinstance(l, list) and len(l) > 0 and isinstance(l[0], list):
        return len(l), len(l[0])
    elif isinstance(l, list):
        return (len(l), )
    else:
        return tuple()

