# Simplified version of the lags, based ONLY on offsets respect the FIRST
# timeslot to predict (with index 1)
#
# There is a difference between past/train lags and future/prediction lags:
#
#   - past/train        lags    refer to past:   they start with 0
#   - future/prediction lags    refer to future: they start with 1  (-1 to be correct)
#
# Note: is compatible with sktime cutoff/ForecastHorizon
#

__all__ = [
    'yx_lags',
    't_lags',
    'tu_lags',
    'to_lags'
]

from typing import Union

import numpy as np
import pandas as pd

from ..utils import RangeType, NoneType


# ---------------------------------------------------------------------------
#   Test matrix
# ---------------------------------------------------------------------------

def dataframe(n, m=0, o=0, *, name="c") -> Union[pd.Series, pd.DataFrame]:
    mat = matrix(n, m, o)
    if len(mat.shape) == 1:
        return pd.Series(data=mat, name=name)
    else:
        columns = [f"{name}{i+1}" for i in range(mat.shape[1])]
        return pd.DataFrame(mat, columns=columns)


def matrix(n, m=0, o=0) -> np.ndarray:
    if m in [None, 0]:
        return np.arange(o+1, o+n+1).astype(int)
    if m == 1:
        return np.arange(o + 1, o + n + 1).astype(int).reshape((-1, 1))

    # if m==1:
    #     return np.arange(1, n+1).reshape((-1,1)).astype(float)

    f = 1
    if m < 10:
        f = 10
    elif m < 100:
        f = 100
    elif m < 1000:
        f = 1000
    else:
        f = 10000

    mat = np.zeros((n,m), dtype=int)
    for i in range(n):
        for j in range(m):
            mat[i,j] = (o+i+1)*f + (j+1)

    return mat


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
# yx_lags
#  t_lags
# tu_lags
# ---------------------------------------------------------------------------

def yxu_lags(lags) -> tuple[list[int], list[int], list[int]]:
    if isinstance(lags, int):
        ylags = lags
        xlags = lags
        ulags = 0
    elif isinstance(lags, dict):
        ylags = lags
        xlags = lags
        ulags = 0
    elif len(lags) == 1:
        ylags = lags[0]
        xlags = 0
        ulags = 0
    elif len(lags) == 2:
        ylags = lags[0]
        xlags = lags[1]
        ulags = 0
    elif len(lags) == 3:
        ylags = lags[0]
        xlags = lags[1]
        ulags = lags[2]
    else:
        raise ValueError(f"Unsupported lags {lags}")

    ylags = to_lags(ylags, 0)
    xlags = to_lags(xlags, 0)
    ulags = to_lags(ulags, 1)
    return ylags, xlags, ulags


def yx_lags(lags) -> tuple[list[int], list[int]]:
    """
    Convert the input lags specification in two list

        - the first one for y
        - the second one for X

    Note: the FIRST last is for 'y', the second one for 'X'

    :param lags: lags for y/x

            l               ->  equivalent to (l, l)
            {...}           ->  equivalent to ({...}, {...})
            []              ->  [[],[]]
            [l,]            ->  [l, []]]
            (ylags, xlags)  ->  [ylags, xlags]

    :return: 2-tuple of lists
    """
    if isinstance(lags, int):
        ylags = lags
        xlags = lags
    elif isinstance(lags, dict):
        ylags = lags
        xlags = lags
    elif len(lags) == 1:
        ylags = lags[0]
        xlags = 0
    elif len(lags) == 2:
        ylags = lags[0]
        xlags = lags[1]
    else:
        raise ValueError(f"Unsupported lags {lags}")

    ylags = to_lags(ylags, 0)
    xlags = to_lags(xlags, 0)
    return ylags, xlags
# end


def t_lags(lags) -> list[int]:
    """
    Convert the lags for the target in a list of integers

    :param lags: lags for t)arget
            None    -> []
            0       -> []
            l       -> [1,...n]
            [...]   -> as is

    :return: list  of lags
    """
    return to_lags(lags, 1)
# end


def tu_lags(lags) -> tuple[list[int], list[int]]:
    if isinstance(lags, int):
        tlags = lags
        ulags = 0
    elif len(lags) == 0:
        tlags = 1
        ulags = 0
    elif len(lags) == 1:
        tlags = lags[0]
        ulags = 0
    elif len(lags) == 2:
        tlags = lags[0]
        ulags = lags[1]
    else:
        raise ValueError(f"Unsupported lags {lags}")

    tlags = to_lags(tlags, 1)
    ulags = to_lags(ulags, 1)
    return tlags, ulags
# end


def to_lags(lags, start=0):
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
    elif isinstance(lags, np.ndarray):
        return lags.tolist()
    elif isinstance(lags, dict):
        return _lags_dict(lags)
    elif isinstance(lags, RangeType):
        return list(lags)
    else:
        raise ValueError(f"Unsupported lag{lags}")
# end


def _lags_dict(d: dict) -> list[int]:
    # {"1": 4}          -> [0,1,2,3]
    # {"1": 7, "7": 3}  -> [0, 1, 2, 3, 4, 5, 6, 7, 14, 21]

    lags: set[int] = set()
    for key, count in d.items():
        f = int(key) if key not in ["", "0", "null"] else 1
        if isinstance(count, int):
            o, c = 0, count
        elif isinstance(count, (tuple, list)) and len(count) == 2:
            o, c = count
        else:
            raise ValueError(f"Unsupported lag {key}:{count}")

        elags = {f*(o + i) for i in range(c)}
        lags = lags.union(elags)
    return list(lags)
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
