#
# Simplified version of the lags, based  ONLY on offsets respect the FIRST
# timeslot to predict (with index 0)
#
# There is a difference between past lags and future lags:
#
#   - past_lags   refer to past:   they start with 1
#   - future_lags refer to future: they start with 0

__all__ = [
    'yx_lags',
    't_lags'
]

from typing import Union

import numpy as np
from ..utils import RangeType, NoneType


# ---------------------------------------------------------------------------
# compute_input_output_shapes
# ---------------------------------------------------------------------------

def compute_input_output_shapes(
    X: np.ndarray,
    y: np.ndarray,
    xlags: Union[list[int], RangeType],
    ylags: Union[list[int], RangeType],
    tlags: Union[list[int], RangeType]
) -> tuple[tuple[int, int], tuple[int, int]]:

    sx = len(xlags) if X is not None else []
    sy = len(ylags)
    st = len(tlags)

    mx = X.shape[1] if X is not None and sx > 0 else 0
    my = y.shape[1]

    return (sy, mx + my), (st, my)
# end


# ---------------------------------------------------------------------------
# tlags_start
# lmax
# ---------------------------------------------------------------------------

def tlags_start(tlags: list[int]):
    n = len(tlags)
    for i in range(n):
        if tlags[i] >= 0:
            return i
    raise ValueError(f"Parameter 'tlags' doesnt' contain positive timeslots: {tlags}")


def lmax(l: list[int]) -> int:
    """
    Maximum value in a list, 0 for empty lists
    """
    return 0 if len(l) == 0 else max(l)


# ---------------------------------------------------------------------------
# yx_lags
#  t_lags
# ---------------------------------------------------------------------------

def yx_lags(lags) -> tuple[list[int], list[int]]:
    """
    Convert the inpyt lags specification in two list

        - the first one for y
        - the second one for X

    Note: the FIRST last is for 'y', the second one for 'X'

    :param lags: lags for y/x

            l               ->  equivalent to (l, l)
            []              ->  [[],[]]
            [l,]            ->  [l, []]]
            (ylags, xlags)  ->  [ylags, xlags]
    :return: 2-tuple of lists
    """
    if isinstance(lags, int):
        ylags = _tolags(lags, 1)
        xlags = ylags
    elif isinstance(lags, (list, tuple)):
        if len(lags) == 0:
            ylags, xlags = [], []
        elif len(lags) == 1:
            ylags = _tolags(lags[0], 1)
            xlags = []
        elif len(lags) == 2:
            ylags = _tolags(lags[0], 1)
            xlags = _tolags(lags[1], 1)
        else:
            raise ValueError(f"Unsupported lags type {type(lags)}")
    else:
        raise ValueError(f"Unsupported lags type {type(lags)}")
    return ylags, xlags
# end


def t_lags(lags) -> list[int]:
    """
    Convert the lags for the target in a list of integers

    :param lags: lags for t)arget
            l       -> [0,1,...n-1]
            [...]   -> as is
    :return: list  of lags
    """
    return _tolags(lags, 0)
# end


def _tolags(lags, start):
    """
    Convert 'lags' in a list of integers
    :param lags: x/y lags specification
            None            -> []
            l               -> [start, ... start+l-1]
            [l1, ...]       -> as is
            (l1, ...)       -> as list
            range(...)      -> expand as list

    :param start: start value
            1   for x/y lags
            0   for t)arget lags

    :return: lags as list of integers
    """
    if isinstance(lags, NoneType):
        return []
    elif isinstance(lags, list):
        return lags
    elif isinstance(lags, int):
        return list(range(start, lags+start))
    elif isinstance(lags, tuple):
        return list(lags)
    elif isinstance(lags, RangeType):
        return list(lags)
    else:
        raise ValueError(f"Unsupported lags type {type(lags)}")
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
