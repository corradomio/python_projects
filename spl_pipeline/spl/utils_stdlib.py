from typing import Iterable


def list_map(f, l):
    return list(map(f, l))


def list_filter(p, l):
    r = list()
    for e in l:
        if p(e):
            r.append(e)
    return r


def set_filter(p, l):
    r = set()
    for e in l:
        if p(e):
            r.add(e)
    return r


def flatten(ll: Iterable) -> list:
    """
    Flatten a list fo lists in a single list
    :param ll:
    :return: the flattened list
    """
    f = []
    for l in ll:
        f += l
    return f
# end


def is_string_list(l) -> bool:
    return isinstance(l, list) and all(map(lambda s: isinstance(s, str), l))
