
def dict_get(d:dict, keys:list[str], defval):
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
    if k not in c:
        return defval
    else:
        return c[k]
# end

