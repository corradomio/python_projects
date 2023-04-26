from typing import Any, Union, Optional
from path import Path as path
from datetime import datetime


NoneType = type(None)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_isinstance_list(l, etype):
    assert isinstance(l, (list, tuple))
    for e in l:
        assert isinstance(e, etype)


def assert_isinstance_dict(d, ktype, vtype):
    assert isinstance(d, dict)
    for k in d.keys():
        v = d[k]
        assert isinstance(k, ktype)
        assert isinstance(v, vtype)


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def module_path():
    import sys
    this_path = path(sys.modules[__name__].__file__)
    return this_path.parent


# def flatten(l: list) -> list:
#     if not isinstance(l, (list, tuple)):
#         return [l]
#     else:
#         f = []
#         for e in l:
#             f += flatten(e)
#         return f
# # end

def tobool(s: str) -> bool:
    if s in [0, False, '', 'f', 'false', 'F', 'False', 'FALSE', 'off', 'no', 'close']:
        return False
    if s in [1, True, 't', 'true', 'T', 'True', 'TRUE', 'on', 'yes', 'open']:
        return True
    else:
        raise ValueError(f"Unsupported boolean value '{s}'")


def qualified_name(o: Any):
    klass = o.__class__
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__   # avoid outputs like 'builtins.str'
    return module + '.' + klass.__qualname__
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


def qualified_name(clazz: type) -> str:
    return f'{clazz.__module__}.{clazz.__name__}'


# ---------------------------------------------------------------------------
# generic utilities
# ---------------------------------------------------------------------------

def kwval(kwargs: dict[str, Any], key: str, defval: Any = None) -> Any:
    """
    Return the value in the dictionary with key 'name' or the default value

    :param kwargs: dictionary containing pairs (key, value)
    :param key: key to read
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


def dict_exclude(d1: dict, keys: Union[None, str, list[str]]) -> dict:
    """
    Remove from the dictionary some keys (and values
    :param d1: dictionary
    :param keys: keys to remove
    :return: the new dictionary
    """
    # keys is a string
    if keys is None:
        return {} | d1
    if isinstance(keys, str): keys = [keys]
    assert isinstance(keys, (list, tuple))

    # keys is None/empty
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
# generic utilities
# ---------------------------------------------------------------------------

# def dict_numpy_rename_1(data: dict, best_name: str, columns: list[str]):
#     assert len(columns) == 1
#     renamed = {}
#     for model_name in data:
#         if model_name == best_name:
#             prefix = ""
#         else:
#             prefix = model_name + ":"
#         values = data[model_name]
#         if len(values.shape) == 2 and values.shape[1] == 1:
#             values = values.reshape(values.shape[0])
#         key = prefix + columns[0]
#         renamed[key] = values
#     # end
#     return renamed
# # end


# def dict_numpy_rename_n(data: dict, best_name: str, columns: list[str]):
#     n = len(columns)
#     renamed = {}
#     for model_name in data:
#         if model_name == best_name:
#             prefix = ""
#         else:
#             prefix = model_name + ":"
#         values = data[model_name]
#         for i in range(n):
#             key = prefix + columns[i]
#             v = values[:, i]
#             renamed[key] = v
#         # end
#     # end
#     return renamed
# # end


# ---------------------------------------------------------------------------
# is_filesystem
# ---------------------------------------------------------------------------

# def datasource_normalize(datasource: Optional[str]) -> Optional[str]:
#     if datasource is None:
#         return None
#     else:
#         return datasource.replace('\\', '/')


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
    assert (dt, (type(None), str))

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


def autoparse_bool(b: Union[None, bool, str]) -> bool:
    if b in [None, 0, False, "false", "False", "FALSE", "f", "F", "off", "close"]:
        return False
    if b in [1, True, "true", "True", "TRUE", "t", "T", "on", "open"]:
        return True
    else:
        raise ValueError(f"Unsuported boolean value '{b}'")
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
