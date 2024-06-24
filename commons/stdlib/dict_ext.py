from typing import Union, Optional

from deprecated import deprecated

from .convert import as_list, CollectionType


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
