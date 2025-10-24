import numpy as np
from ..utils import is_instance


# ---------------------------------------------------------------------------
# tlags_start
# lmax
# reverse
# ---------------------------------------------------------------------------

def t_start(tlags: list[int]):
    n = len(tlags)
    for i in range(n):
        if tlags[i] >= 0:
            return i
    raise ValueError(f"Parameter 'tlags' doesnt' contain positive timeslots: {tlags}")


def t_step(lags: list[int]) -> int:
    """
    Compute the maximum advance step it is possible to do.
    If [1], the step is 1
    If [1,2,3] the step is 3
    If [1,3], the step is 1
    """
    assert is_instance(lags, list[int]) and len(lags) > 0

    n = len(lags)
    if n == 1:
        return 1
    elif lags[0] == 1 and lags[-1] == n:
        return n
    else:
        return 1
# end


def lmax(l: list[int]) -> int:
    """
    Maximum value in a list, 0 for empty lists
    """
    return 0 if len(l) == 0 else max(l)


def reverse(l):
    return list(reversed(l))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _transpose(t):
    return None if t is None else t.swapaxes(1, 2)


def _flatten(t, n):
    return None if t is None else t.reshape(n, -1)


def _concat(*tlist):
    # remove None values
    tlist = [t for t in tlist if t is not None]
    if len(tlist) == 0:
        return None
    if len(tlist) == 1:
        return tlist[0]
    else:
        return np.concatenate(tlist, axis=-1)
