from typing import Iterable, Union, Any
from .types import *


def as_list(obj: Union[NoneType, str, list, tuple], param=None):
    """
    Convert parameter 'obj' in a list.
    If 'obj' is None, the empty list, if a string|int|..., in a singleton list

    :param obj: value to convert
    :param param: parameter's name, used in the error message
    :return: a list
    """
    tl = type(obj)
    assert tl in (NoneType, int, str, list, tuple), f"'{param}' not of type None, int, str, list, tuple"
    if tl is NoneType:
        return []
    elif tl == list:
        return obj
    elif tl == tuple:
        return list(obj)
    else:
        return [obj]
    # return [] if l is None else \
    #         [l] if tl == str else \
    #         list(l) if tl == tuple else l


def as_tuple(obj: Union[NoneType, Any, list, tuple], param=None):
    tl = type(obj)
    assert tl in (NoneType, str, list, tuple), f"'{param}' not of type None, str, int, list, tuple"
    if tl is NoneType:
        return ()
    elif tl == tuple:
        return obj
    elif tl == list:
        return tuple(obj)
    else:
        return (obj,)


def as_dict(d: Union[NoneType, dict], *, key=None) -> dict:
    #
    # also   d or {}
    if isinstance(d, dict):
        return d
    if d is None:
        return dict()
    elif key is not None:
        return {key: d}
    else:
        raise ValueError(f"Unsupported value type: {type(d)}")


# ---------------------------------------------------------------------------
# tobool
# ---------------------------------------------------------------------------

def to_bool(s: str) -> bool:
    """
    Convert the value (can be a boolean value, an integer or a string) into a boolean value.
    Supported conversions:

        False: 0, False, '', 'f', 'false', 'off', 'no',  'close'
         True: 1, True,      't', 'true',  'on',  'yes', 'open'

    Note: it is a 'better' version of the Python's boolean conversion rules, because
    it is based on a 'real boolean value' represented in different ways.
    However it supports some 'extended' values:

        None            -> False
        integer != 0    -> True

    :param s: a string or other compatible value
    :return: boolean value
    """
    if s is None:
        return False
    if isinstance(s, str):
        s = s.lower()
    if s in [0, False, '', 'f', 'false', 'off', 'no', 'close', '0']:
        return False
    if s in [1, True, 't', 'true', 'on', 'yes', 'open', '1']:
        return True
    if isinstance(s, int):
        return s != 0
    else:
        raise ValueError(f"Unsupported boolean value '{s}'")


# Alias
tobool = to_bool


# ---------------------------------------------------------------------------
# to_float
# ---------------------------------------------------------------------------

def to_float(x) -> Union[float, list[float]]:
    """
    Convert, recursively, each object in a float:
    1) int -> float
    2) str -> float
    3) collection -> list of floats
    """
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [float(e) for e in x]
    if isinstance(x, Iterable):
        return list(map(lambda t: to_float(t), x))
    else:
        return float(x)
# end

# Alias
tofloat = to_float

