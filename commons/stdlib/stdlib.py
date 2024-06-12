from deprecated import deprecated
from typing import Any, Union, Optional, Iterable
from path import Path as path

NoneType = type(None)
RangeType = type(range(0))
CollectionType = (list, tuple)
FunctionType = type(lambda x: x)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

# def to_name(ts) -> str:
#     if len(ts) == 0:
#         return "root"
#     else:
#         return "/".join(list(ts))


# def from_name(tsname) -> tuple[str]:
#     if tsname == 'root':
#         return tuple()
#     else:
#         return tuple(tsname.split('/'))


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

def module_path():
    """
    Pyhon module of the current class
    """
    import sys
    this_path = path(sys.modules[__name__].__file__)
    return this_path.parent
# end


def qualified_type(value: Any) -> str:
    """
    Fully qualified type of the specified value.
    """
    return qualified_name(type(value))


def qualified_name(klass: Any) -> str:
    """
    Fully qualified of the class.
    For builtin classes, only the name
    """
    if isinstance(klass, str):
        return klass
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__   # avoid outputs like 'builtins.str'
    return f'{module}.{klass.__qualname__}'
# end


def import_from(qname: str) -> Any:
    """
    Import a class specified by the fully qualified name string

    :param qname: fully qualified name of the class
    :return: Python class
    """
    import importlib
    p = qname.rfind('.')
    qmodule = qname[:p]
    name = qname[p+1:]

    module = importlib.import_module(qmodule)
    clazz = getattr(module, name)
    return clazz
# end


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def list_map(f, l):
    """
    Same as 'list(map(f, l))'
    """
    return list(map(f, l))


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
        return [obj]
    # return tuple() if l is None else \
    #     l if tl == tuple else \
    #     tuple(l) if tl == list else (l,)


def as_dict(d: Union[NoneType, dict]) -> dict:
    return dict() if d is None else d


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


# ---------------------------------------------------------------------------
# Keyword parameters
# ---------------------------------------------------------------------------
#
#   - ricuperare il valore di un parametro dato il nome ed il valore
#     di default
#   - estrarre i parametri che hanno un certo prefisso, seguito da "__"
#     usato in skorch, ad esempio.
#
#


def kwval(kwargs: dict[Union[str, tuple], Any], key: Union[None, str, tuple, list] = None, defval: Any = None, keys=None) -> Any:
    """
    Return the value in the dictionary with key 'name' or the default value

    :param kwargs: dictionary containing pairs (key, value)
    :param key: key (or alternative names) to read
    :param keys: list of keys used to navigate the dictionary
    :param defval: value to return is the key is not in the dictionary
    :return: the value in the dictionary or the default value
    """
    def _parse_val(val):
        if not isinstance(defval, str) and isinstance(val, str):
            if defval is None:
                return val
            if isinstance(defval, bool):
                return tobool(val)
            if isinstance(defval, int):
                return int(val)
            if isinstance(defval, float):
                return float(val)
            else:
                raise ValueError(f"Unsupported conversion from str to '{type(defval)}'")
        return val

    if keys is not None:
        assert isinstance(keys, CollectionType)
        n = len(keys)
        for i in range(n-1):
            key = keys[i]
            if key not in kwargs:
                return defval
            kwargs = kwargs[key]
        # end
        key = keys[-1]
        if key not in kwargs:
            return defval
        else:
            return _parse_val(kwargs[key])

    if isinstance(key, CollectionType):
        altkeys = key
        for key in altkeys:
            if key in kwargs:
                return _parse_val(kwargs[key])
        return defval
    elif key in kwargs:
        return _parse_val(kwargs[key])
    else:
        return defval


def kwparams(kwargs: dict, prefix: str) -> dict:
    """
    Extract the parameters with prefix '<prefix>__<name>' returning
    a dictionary containing the parameters with name '<name>'

    Example:

        d = {
            'criterion': ...,
            'criterion__beta': 1,
            'criterion__sigma': sigma
        }

        kwparams(d, 'criterion') -> {
            'beta': 1,
            'sigma': sigma
        }

    :param kwargs: keyword parameters
    :param prefix: prefix to use
    :return:
    """
    p = f"{prefix}__"
    l = len(p)

    params = {}
    for kw in kwargs:
        if kw.startswith(p):
            n = kw[l:]
            params[n] = kwargs[kw]
    return params


def kwselect(kwargs: dict, prefix: str) -> dict:
    """
    Select the parameters with the specified prefix.
    The keys are no changed as in 'kwparams()'

    :param kwargs: keyword parameters
    :param prefix: prefix to use
    :return:
    """
    selected = {}
    for kw in kwargs:
        if kw.startswith(prefix):
            selected[kw] = kwargs[kw]
    return selected


def kwexclude(kwargs: dict, exclude: Union[str, list[str]]) -> dict:
    """
    Create a new dictionary without keys having as prefix a string in 'exclude'

    :param kwargs: keyword parameters
    :param keys: prefix(es) to exclude
    :return: a new dictionary without the excluded parameters
    """
    exclude = as_list(exclude, 'exclude')

    def has_prefix(k: str):
        for prefix in exclude:
            # p = f"{prefix}__"
            p = prefix
            if k.startswith(p):
                return True
        return False

    filtered = {}
    for kw in kwargs:
        if not has_prefix(kw):
            filtered[kw] = kwargs[kw]

    return filtered


# ---------------------------------------------------------------------------
# is_filesystem
# ---------------------------------------------------------------------------

def is_filesystem(url: str) -> bool:
    """
    Check if the url is a filesystem, sthat is, starting with 'file://' or '<disk>:'.
    It must have the form '<protocol>://<rest>' or '<disk>:<rest>'

    :param url: url to analyze
    :return: true if it is a filesystem url
    """
    # file://....
    # <disk>:....
    if url.startswith("file://") or len(url) > 2 and url[1] == ':':
        return True
    elif "://" in url:
        return False
    else:
        raise ValueError(f"Unsupported datasource '{url}'")


# ---------------------------------------------------------------------------
# dict utilities
# ---------------------------------------------------------------------------

@deprecated(reason='Supported by stdlib.dict.contains_keys()')
def dict_contains_some(d: dict, keys: Union[str, list[str]]):
    """
    Check if the dictionary contains some key in the list

    :param d: dictionary
    :param keys: key name or list of names
    """
    keys = as_list(keys, "keys")
    for k in keys:
        if k in d:
            return True
    return False


@deprecated(reason="Supported by builtin.dict '|' operator:  d1|d2")
def dict_union(d1: dict, d2: dict, inplace=False) -> dict:
    """
    Union of 2 dictionaries. The second dictionary will override the
    common keys in the first one.

    :param d1: first dictionary
    :param d2: second dictionary
    :return: merged dictionary (a new one)
    """
    if inplace:
        for k in d2:
            d1[k] = d2[k]
        return d1
    else:
        return {} | d1 | d2


@deprecated(reason="Supported by stdlib.dict.select(mode='select')")
def dict_select(d: dict, keys: list[str]) -> dict:
    s = {}
    for k in keys:
        if k in d:
            s[k] = d[k]
    return s


@deprecated(reason="Supported by stdlib.dict.select(mode='exclude')")
def dict_exclude(d1: dict, keys: Union[None, str, list[str]]) -> dict:
    """
    Remove from the dictionary some keys (and values
    :param d1: dictionary
    :param keys: keys to remove
    :return: the new dictionary
    """
    # keys is a string
    if keys is None: keys = []
    if isinstance(keys, str): keys = [keys]
    assert isinstance(keys, CollectionType)

    # keys is None/empty
    if len(keys) == 0:
        return d1
    # dict doesn't contain keys
    if len(set(keys).intersection(d1.keys())) == 0:
        return d1

    d = {}
    for k in d1:
        if k in keys:
            continue
        d[k] = d1[k]
    return d


def dict_rename(d: dict, k1: Union[str, list[str], dict[str, str]], k2: Optional[str]=None) -> dict:
    """
    Rename the key 'k1' in the dictionary as 'k2
    :param d: dictionary
    :param k1: key to rename, or a list of tuples [(kold, knew), ...)
                or a dict {kold: knew, ...}
    :param k2: new name to use
    :return: the new dictionary
    """
    def _renk(kold, knew):
        if kold in d:
            v = d[kold]
            del d[kold]
            d[knew] = v

    if isinstance(k1, str):
        _renk(k1, k2)
    elif isinstance(k1, CollectionType):
        klist = k1
        for k1, k2 in klist:
            _renk(k1, k2)
    elif isinstance(k1, dict):
        kdict = k1
        for k1 in kdict:
            k2 = kdict[k1]
            _renk(k1, k2)
    return d


@deprecated(reason='Supported by stdlib.dict.delete_keys()')
def dict_del(d: dict, keys: Union[str, list[str]]) -> dict:
    """
    Remove the list of keys from the dictionary
    :param d: dictionary
    :param keys: key(s) to remove
    :return: the updated dictionary
    """
    if isinstance(keys, str):
        keys = list[keys]
    for k in keys:
        if k in d:
            del d[k]
    return d


@deprecated(reason='Supported by stdlib.dict.to_list()')
def dict_to_list(d: Union[dict, list, tuple]) -> list:
    """
    Convert a dictionary in a list of tuples

        {k: v, ...} -> [(k, v), ...]

    :param d: dictionary, or list or tuple
    :return: the dictionary converted in a list of tuples
    """
    assert isinstance(d, (dict, list, tuple))
    if isinstance(d, (list, tuple)):
        return d
    if len(d) == 0:
        return []

    l = []
    for k in d.keys():
        l.append((k, d[k]))
    return l


# ---------------------------------------------------------------------------
# Real comparisons with error
# ---------------------------------------------------------------------------
# The following comparison predicates can be used for float values where it
# is not possible to do 'safe' comparisons without consider rounding/accumulating
# errors. In this case, it is possible to specify an 'eps' (that can be 'zero')
# to absorb these errors.

EPS: float = 1.e-6


def sign(x, zero=False, eps: float = EPS) -> int:
    """
    Sign of the number:

        -1 if in range (-inf, -eps)
         0 if in range [-eps, +eps]
        +1 if in range (+eps, +inf)

    Note that 'eps' can be 0, in this case 'sign' is 0 only for exactly 0 (zero)

    :param x: value to analyze
    :param zero: if to return 0 (True) or 1 (False) for 'zero values'
    :param eps: values interval to consider 'zero'
    :return: the integer values -1, 0, 1, based on 'x' value
    """
    if x < -eps: return -1
    if x > +eps: return +1
    return 0 if zero else 1


def zero(x, eps: float = EPS) -> float:
    """return 0 if the value is smaller than an eps"""
    return 0. if -eps <= x <= +eps else x


def isz(x: float, eps: float = EPS) -> bool:
    """is zero"""
    return -eps <= x <= eps


def isnz(x: float, eps: float = EPS) -> bool:
    """is not zero"""
    return not isz(x, eps=eps)


def iseq(x: float, y: float, eps: float = EPS) -> bool:
    """is equal to"""
    return isz(x - y, eps=eps)


def isgt(x: float, y: float, eps: float = EPS) -> bool:
    """is greater than"""
    return x > (y + eps)


def islt(x: float, y: float, eps: float = EPS) -> bool:
    """is less than"""
    return x < (y - eps)


def isge(x: float, y: float, eps: float = EPS) -> bool:
    """is greater or equal than"""
    return not islt(x, y, eps=EPS)


def isle(x: float, y: float, eps: float = EPS) -> bool:
    """is less or equal than"""
    return not isgt(x, y, eps=EPS)


# ---------------------------------------------------------------------------
# Simple mathematica function
# ---------------------------------------------------------------------------
from math import sqrt       # DON'T REMOVE!!!


# inverse with check
def inv(x: float, eps: float = EPS) -> float:
    """
    Inverse of the number with check for zero.
    If x is zero, return zero

    :param x: value
    :param eps: epserance
    :return: 1/x or 0
    """
    return 0. if isz(x, eps=eps) else 1. / x


# Square
def sq(x: float) -> float: return x * x

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
