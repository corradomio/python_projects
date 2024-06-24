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
from typing import Union, Any
from .convert import tobool, CollectionType, as_list


def kwval(kwargs: dict[Union[str, tuple], Any],
          key: Union[None, str, tuple, list] = None,
          defval: Any = None) -> Any:
    """
    Return the value in the dictionary with key 'name' or the default value

    :param kwargs: dictionary containing pairs (key, value)
    :param key: key or list of keys used to navigate the dictionary
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

    assert kwargs is not None, "Missing dictionary"
    assert isinstance(key, (str, list, tuple)), "Missing key"

    if isinstance(key, str):
        return _parse_val(kwargs.get(key, defval))

    keys = list(key)
    n = len(keys)
    for i in range(n-1):
        key = keys[i]
        if key not in kwargs:
            return defval
        kwargs = kwargs[key]
    # last key
    key = keys[-1]
    return _parse_val(kwargs.get(key, defval))


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

    [2024/06/18] added compatibility with 'dict_select'

    :param kwargs: keyword parameters
    :param prefix: prefix to use
    :return:
    """
    s = {}
    if isinstance(prefix, str):
        for kw in kwargs:
            if kw.startswith(prefix):
                s[kw] = kwargs[kw]
    else:
        keys = prefix
        d = kwargs
        for k in keys:
            if k in d:
                s[k] = d[k]
        return s
    return s


def kwexclude(kwargs: dict, exclude: Union[str, list[str]]) -> dict:
    """
    Create a new dictionary without keys having as prefix a string in 'exclude'

    :param kwargs: keyword parameters
    :param keys: prefix(es) to exclude
    :return: a new dictionary without the excluded parameters
    """
    assert isinstance(kwargs, dict)
    assert isinstance(exclude, (str, list))

    exclude = as_list(exclude, 'exclude')

    def has_prefix(k: str):
        for p in exclude:
            if k.startswith(p): return True
        return False

    filtered = {}
    for kw in kwargs:
        if not has_prefix(kw):
            filtered[kw] = kwargs[kw]

    return filtered


def kwmerge(kwargs: dict, *kwrest) -> dict:
    # to avoid changes on the original dict
    kwargs = {} | kwargs

    for kw in kwrest:
        if kw is not None:
            for k in kw:
                if k not in kwargs:
                    kwargs[k] = kw[k]
                elif kw[k] is None:
                    pass
                else:
                    kwargs[k] = kw[k]
    # end for/if/for
    return kwargs


