from typing import Any, Union, Optional

from path import Path as path

NoneType = type(None)
RangeType = type(range(0))
CollectionType = (list, tuple)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def to_name(ts) -> str:
    if len(ts) == 0:
        return "root"
    else:
        return "~".join(list(ts))


def from_name(tsname) -> tuple[str]:
    if tsname == 'root':
        return tuple()
    else:
        return tuple(tsname.split('/'))


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


def qualified_name(klass: Any):
    """
    Fully qualified of the class.
    For builtin classes, only the name
    """
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


def as_list(l: Union[NoneType, str, list[str], tuple[str]], param=None):
    """
    Convert parameter 'l' in a list.
    If 'l' is None, the empty list, if a string, in a singleton list

    :param l: value to convert
    :param param: parameter's name, used in the error message
    :return: a list
    """
    tl = type(l)
    assert tl in (NoneType, str, list, tuple), f"'{param}' not of type None, str, list[str]"
    return [] if l is None else \
            [l] if tl == str else \
            list(l) if tl == tuple else l


# ---------------------------------------------------------------------------

def tobool(s: str) -> bool:
    """
    Convert the string into a boolean value.
    Supported conversions:

        False: 0, False, '', 'f', 'false', 'F', 'False', 'FALSE', 'off', 'no',  'close'
         True: 1, True,      't', 'true',  'T', 'True',  'TRUE',  'on',  'yes', 'open'

    :param s: string
    :return: boolean value
    """
    if isinstance(s, str):
        s = s.lower()
    if s in [0, False, '', 'f', 'false', 'off', 'no', 'close', '0']:
        return False
    if s in [1, True, 't', 'true', 'on', 'yes', 'open', '1']:
        return True
    else:
        raise ValueError(f"Unsupported boolean value '{s}'")


def lrange(start, stop=None, step=1) -> list[int]:
    """As range but it returns a list"""
    if stop is None:
        return list(range(start))
    else:
        return list(range(start, stop, step))
# end


# ---------------------------------------------------------------------------
# Numerical aggregation functions
# ---------------------------------------------------------------------------
# sum_  as Python 'sum([...])'
# mul_  as Pythin 'sum([...]) but for multiplication
#

def sum_(x):
    if x is None:
        return 0
    elif isinstance(x, (int, float)):
        return x
    else:
        return sum(x)


def mul_(x):
    if x is None:
        return 1
    elif isinstance(x, (int, float)):
        return x
    else:
        m = 1
        for e in x:
            m *= e
        return m


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

def as_kwargs(locals):
    kwargs = {} | locals
    for key in ['forecaster', 'window_length', 'reduction_strategy',
                'self', '__class__']:
        del kwargs[key]
    return kwargs


def kwval(kwargs: dict[Union[str, tuple], Any], key: Union[None, str, tuple, list] = None, defval: Any = None, keys=None) -> Any:
    """
    Return the value in the dictionary with key 'name' or the default value

    :param kwargs: dictionary containing pairs (key, value)
    :param key: key (or alternate names) to read
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
    # end

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
    a dictionary using '<name>'

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
    sep = "__"
    p = f"{prefix}{sep}"
    l = len(p)

    params = {}
    for kw in kwargs:
        if kw.startswith(p):
            n = kw[l:]
            params[n] = kwargs[kw]
    return params

kwselect = kwparams


def kwexclude(kwargs: dict, exclude: Union[str, list[str]]) -> dict:
    """
    Create a new dictionary without keys having as prefix a string in exclude
    :param kwargs:
    :param keys:
    :return:
    """
    exclude = as_list(exclude, 'exclude')

    def has_prefix(k: str):
        for p in exclude:
            if k.startswith(p):
                return True
        return False

    filtered = {}
    for kw in kwargs:
        if not has_prefix(kw):
            filtered[kw] = kwargs[kw]

    return filtered


# ---------------------------------------------------------------------------
# dict utilities
# ---------------------------------------------------------------------------

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


def dict_select(d: dict, keys: list[str]) -> dict:
    s = {}
    for k in keys:
        if k in d:
            s[k] = d[k]
    return s


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
# is_filesystem
# ---------------------------------------------------------------------------

def is_filesystem(datasource: str) -> bool:
    # file://....
    # <disk>:....
    if datasource.startswith("file://") or len(datasource) > 2 and datasource[1] == ':':
        return True
    elif "://" in datasource:
        return False
    else:
        raise ValueError(f"Unsupported datasource '{datasource}'")


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
