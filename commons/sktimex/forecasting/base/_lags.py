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

from ...utils import RangeType, NoneType


def _tolags(lags, start):
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


def yx_lags(lags) -> tuple[list[int], list[int]]:
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


def t_lags(lags) -> list[int]:
    return _tolags(lags, 0)
