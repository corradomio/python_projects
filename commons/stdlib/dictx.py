
def dict_get(d:dict, keys:list[str], defval=None):
    """
    Scan the dictionary and retrieve the value specified by the
    list of keys, or the default value
    """
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
    s = {}
    for k in d:
        if k in keys:
            s[k] = d[k]
    return s
# end


def dict_exclude(d:dict, keys:list[str]) -> dict:
    """
    Create a dict containing only the keys not in the list
    """
    s = {}
    for k in d:
        if k not in keys:
            s[k] = d[k]
    return s
# end

