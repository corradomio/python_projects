from path import Path as path
from datetime import datetime
from typing import Any, Union, Optional

NoneType = type(None)
CollectionType = (list, tuple)


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

def module_path():
    import sys
    this_path = path(sys.modules[__name__].__file__)
    return this_path.parent


def qualified_name(klass: Any):
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__   # avoid outputs like 'builtins.str'
    return f'{module}.{klass.__qualname__}'
# end


def type_of(obj: object):
    return str(type(obj))


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------
# tuple, sep -> str
# str, sep -> tuple

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
# Generic utilities
# ---------------------------------------------------------------------------

def list_map(f, l):
    """
    Same as 'list(map(f, l))'
    """
    return list(map(f, l))


def lrange(start, stop=None, step=None):
    """
    Same as 'list(range(...))'
    """
    if stop is None:
        return list(range(start))
    elif step is None:
        return list(range(start, stop))
    else:
        return list(range(start, stop, step))


def tobool(s: str) -> bool:
    """
    Convert the string into a boolean value.
    Supported conversions:

        False: 0, False, '', 'f', 'false', 'F', 'False', 'FALSE', 'off', 'no',  'close'
         True: 1, True,      't', 'true',  'T', 'True',  'TRUE',  'on',  'yes', 'open'

    :param s: string
    :return: boolean value
    """
    if s in [0, False, '', 'f', 'false', 'F', 'False', 'FALSE', 'off', 'no', 'close']:
        return False
    if s in [1, True, 't', 'true', 'T', 'True', 'TRUE', 'on', 'yes', 'open']:
        return True
    else:
        raise ValueError(f"Unsupported boolean value '{s}'")


# ---------------------------------------------------------------------------
# as_list
# ---------------------------------------------------------------------------

def as_list(l: Union[NoneType, str, list[str], tuple[str]], param):
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


_as_list = as_list


# ---------------------------------------------------------------------------
# import_from
# ---------------------------------------------------------------------------

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
# generic utilities
# ---------------------------------------------------------------------------

# def as_kwargs(locals):
#     kwargs = {} | locals
#     for key in ['forecaster', 'window_length', 'reduction_strategy',
#                 'self', '__class__']:
#         del kwargs[key]
#     return kwargs
# # end


def kwval(kwargs: dict, key: str, defval: Any = None) -> Any:
    """
    Return the value in the dictionary with key 'name' or the default value

    Note: it convert automatically the string into the same type of defval

    :param kwargs: dictionary containing pairs (key, value)
    :param key: key (or alternate names) to read
    :param defval: value to return is the key is not in the dictionary
    :return: the value in the dictionary or the default value
    """
    if key not in kwargs:
        return defval

    val = kwargs[key]
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


def dict_contains_some(d: dict, keys: Union[str, list[str]]):
    """
    Check if the dictionary contains some key in the list
    """
    keys = as_list(keys, "keys")
    for k in keys:
        if k in d:
            return True
    return False


def dict_union(d1: dict, d2: dict) -> dict:
    """
    Union of 2 dictionaries. The second dictionary will override the
    common keys in the first one.

    :param d1: first dictionary
    :param d2: second dictionary
    :return: merged dictionary (a new one)
    """
    if d1 is None: d1 = {}
    if d2 is None: d2 = {}

    d = {} | d1 | d2
    return d


def dict_update(d1: dict, d2: dict) -> dict:
    """
    Update dictionary 'd1' with the content of 'd2

    :param d1: the dictionary to update
    :param d2: the dictionary used for the updates
    :return: the updated dictionary
    """
    for k in d2:
        d1[k] = d2[k]
    return d1


def dict_select(d: dict, keys: Union[str, list[str]], prefix=None) -> dict:
    """
    Create a new dictionary containing only the keys specified

    Example:

    :param d: original dictionary
    :param keys: keys to select
    :param prefix: prefix used to save the keys in 'd'
    :return: new dictionary
    """
    keys = as_list(keys, "keys")
    sd = {}
    for k in keys:
        # if prefix is defined, try to use <prefix><key>
        # if <prefix><key> is not present, try tp use <key>
        if prefix:
            dk = prefix + k
            if dk not in d:
                dk = k
        else:
            dk = k

        if dk in d:
            sd[k] = d[dk]
    return sd


def dict_exclude(d1: dict, keys: Union[None, str, list[str]]) -> dict:
    """
    Remove from the dictionary some keys (and values

    :param d1: dictionary
    :param keys: keys to remove
    :return: the new dictionary
    """
    keys = as_list(keys)
    # keys is a string
    if len(keys) == 0:
        return {} | d1

    d = {}
    for k in d1:
        if k in keys:
            continue
        d[k] = d1[k]
    return d


def dict_rename(d: dict, k1: str, k2: str) -> dict:
    """
    Rename the key 'k1' in the dictionary as 'k2

    :param d: dictionary
    :param k1: key to rename
    :param k2: new name to use
    :return: the new dictionary
    """
    if k1 not in d:
        return d
    val = d[k1]
    del d[k1]
    d[k2] = val
    return d


def dict_del(d: dict, keys: Union[str, list[str]]) -> dict:
    """
    Remove the list of keys from the dictionary
    :param d: dictionary
    :param keys: key(s) to remove
    :return: the updated dictionary
    """
    d = {} | d
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
# autoparse_datetime
# ---------------------------------------------------------------------------
#   yyyy
#   yyyy/mm             yyyy-mm
#   yyyy/mm/dd          yyyy-mm-dd
#   yyyy/mm/dd HH:MM    yyyy-mm-dd HH:MM
#   yyyy/mm/dd HH:MM:SS yyyy-mm-dd HH:MM:SS
#
#   <date>T<time>
#

def autoparse_datetime(dt: Optional[str]) -> Optional[datetime]:
    # assert (dt, (NoneType, str))

    if dt is None:
        return None

    n_slashs = dt.count('/')
    n_dashes = dt.count('-')
    n_colons = dt.count(':')

    if 'T' in dt:
        dt = dt.replace('T', ' ')

    if n_dashes == 2 and n_colons == 2:
        return datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    if n_dashes == 2 and n_colons == 1:
        return datetime.strptime(dt, '%Y-%m-%d %H:%M')
    if n_dashes == 2 and n_colons == 0:
        return datetime.strptime(dt, '%Y-%m-%d')

    if n_slashs == 2 and n_colons == 2:
        return datetime.strptime(dt, '%Y/%m/%d %H:%M:%S')
    if n_slashs == 2 and n_colons == 1:
        return datetime.strptime(dt, '%Y/%m/%d %H:%M')
    if n_slashs == 2 and n_colons == 0:
        return datetime.strptime(dt, '%Y/%m/%d')

    if n_dashes == 1:
        return datetime.strptime(dt, '%Y-%m')
    if n_slashs == 1:
        return datetime.strptime(dt, '%Y/%m')

    else:
        return datetime.strptime(dt, '%Y')
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
