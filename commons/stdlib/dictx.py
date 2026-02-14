#
#   dict_get(d:dict, keys:list[str], defval=None)
#   dict_select( d:dict, keys:list[str]) -> dict
#   dict_exclude(d:dict, keys:list[str]) -> dict
#   reverse_dict(d:dit) -> dict
#
#   .

from typing import Union


def dict_get(d:dict, keys:list[str], defval=None):
    """
    Scan the dictionary and retrieve the value specified by the
    list of keys, or the default value
    """
    if isinstance(keys, str):
        keys = [keys]

    pkeys = keys[:-1]
    c = d
    for k in pkeys:
        if k not in c:
            return defval
        c = c[k]
        if not isinstance(c, dict):
            return defval
    # end
    k = keys[-1]
    return c[k] if k in c else defval
# end


def dict_select(d:dict, keys:list[str]) -> dict:
    """
    Create a dict containing only the keys in the list
    """
    if isinstance(keys, str):
        keys = [keys]

    s = {}
    for k in d:
        if k in keys:
            s[k] = d[k]
    return s
# end


def dict_exclude(d:dict, keys:list[str]) -> dict:
    """
    Create a dict excluding the keys in the list
    """
    if isinstance(keys, str):
        keys = [keys]

    s = {}
    for k in d:
        if k not in keys:
            s[k] = d[k]
    return s
# end


def reverse_dict(d: Union[dict, dict]) -> dict:
    """
    Create a reversed dictionary

    :param d: dictionary
    :return: the reversed dictionary
    """
    drev = {d[k]: k for k in d}
    assert len(d) == len(drev)
    return drev
# end


def dict_resolve(config: dict, params: dict) -> dict:
    assert isinstance(config, dict)
    assert isinstance(params, dict)

    def _check(name):
        if name not in params:
            raise KeyError(f"Parameter '{name}' not specified")

    def vrepl(v):
        # if isinstance(v, np.integer):
        #     return int(v)
        # if isinstance(v, np.inexact):
        #     return float(v)
        # if isinstance(v, np.bool_):
        #     return bool(v)
        if not isinstance(v, str):
            return v
        # "${<name>}"
        if v.startswith("${") and v.endswith("}"):
            name = v[2:-1]
            _check(name)
            return params[name]
        # "...${<name>..."
        while "${" in v:
            s = v.find("${")
            e = v.find("}", s)
            name = v[s + 2:e]
            _check(name)
            v = v[:s] + str(params[name]) + v[e + 1:]
        # "$<name>"
        if v.startswith("$"):
            name = v[1:]
            _check(name)
            return params[name]
        else:
            return v

    def drepl(d: dict) -> dict:
        # skip the keys starting with '#...'
        return {
            k: repl(d[k])
            for k in d if not k.startswith("#")
        }

    def lrepl(l: list) -> list:
        return [
            repl(v)
            for v in l
        ]

    def repl(v):
        if isinstance(v, dict):
            return drepl(v)
        if isinstance(v, list):
            return lrepl(v)
        else:
            return vrepl(v)

    return repl(config)
# end


