from typing import Any, Optional, Union, cast

import numpy as np


# ---------------------------------------------------------------------------
# module_path
# qualified_type
# qualified_name
# ---------------------------------------------------------------------------

def module_path(name=None) -> str:
    """
    Pyhon module of the current class
    """
    name = __name__ if name is None else name
    import sys
    this_path = sys.modules[name].__file__
    slash = this_path.rfind('/')
    if slash == -1:
        slash = this_path.rfind('\\')
    return this_path[:slash]


def qualified_type(value: Any) -> str:
    """
    Fully qualified type of the specified value.
    """
    return qualified_name(type(value))


def qualified_name(klass: type) -> str:
    """
    Fully qualified of the class.
    For builtin classes, only the name
    """
    if isinstance(klass, str):
        return klass
    module = klass.__module__
    if module == 'builtins':
        return klass.__qualname__   # avoid outputs like 'builtins.str'
    else:
        return f'{module}.{klass.__qualname__}'


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


# ---------------------------------------------------------------------------
# create_from
# ---------------------------------------------------------------------------

def resolve(config: dict, params: dict={}) -> dict:
    assert isinstance(config, dict)
    assert isinstance(params, dict)

    # if len(params) == 0:
    #     return config

    def is_param(v):
        return isinstance(v, str) and v.startswith("{") and v.endswith("}")

    def has_param(v):
        return isinstance(v, str) and "{" in v and "}" in v

    def value_of(v: str):
        p = v[1:-1]
        return params[p]

    def insert_into(v: str):
        while("{" in v):
            bgn = v.find("{")
            end = v.find("}", bgn + 1)
            p = v[bgn+1:end]
            v = v[:bgn] + params[p] + v[end + 1:]
        return v

    def repl(v):
        if isinstance(v, dict):
            return drepl(v)
        if isinstance(v, list):
            return lrepl(v)
        if isinstance(v, str):
            if is_param(v):
                return value_of(v)
            elif has_param(v):
                return insert_into(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.inexact):
            return float(v)
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

    return repl(config)


def create_from(config: str | dict, params: dict={}, aliases: dict={}) -> Any:
    """
    Create and instance of the object.
    If it is a string, it must be the fully qualified class name.
    If it is a dictionary, it must have the form:

    instance_info: {
        "class": "qname|alias|type",
        "p1": <v1>, ...
    }

    It is possible to use 'alias' names, passed in the extra parameter 'aliases

    aliases: {
        "alias": "qname|type"
    }
    
    Note: the class can be a Python class object
    
    """
    if config is None:
        return None

    if isinstance(config, (str, type)):
        config = {"class": config}

    assert isinstance(config, dict)
    assert isinstance(aliases, dict)
    assert isinstance(params, dict)

    config = resolve(config, params)

    qname: str = ""
    clazz_args = {} | config

    # delete '#name' parameters
    # for k in config:
    #     if k.startswith("#"):
    #         del clazz_args[k]

    # delete one of the 'class' keywords
    for k in ["class", "class_name", "clazz", "method"]:
        if k in config:
            qname = config[k]
            if qname in aliases:
                qname = aliases[qname]
            del clazz_args[k]
            break

    assert isinstance(qname, str) and len(qname) > 0 or isinstance(qname, type), \
        "Missing mandatory key: ('class', 'class_name', 'clazz', 'method')"
    if isinstance(qname, type):
        clazz = qname
    else:
        clazz = cast(type, import_from(qname))
    return clazz(**clazz_args)


def create_from_collection(collection: Union[None, list, dict]) -> Union[None, list, dict]:
    """
    Create a list of objects or a dictionary of objects.
    The list must have the form:

        [{"class": ..., ...}, ...]

    The dictionary must have the form:

        {
            "name": {
                "class": ...
                ...
            },
            ...
        }

    """
    if collection is None:
        return None

    if isinstance(collection, dict):
        return {
            name: create_from(config)
            for name, config in collection.items()
        }
    elif isinstance(collection, list):
        return [
            create_from(config)
            for config in collection
        ]
    else:
        raise ValueError(f"Unsupported collection type {type(collection)}")


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

def namespace_of(s: str) -> str:
    """First component of a fully qualified name"""
    p = s.find('.')
    return s[:p]


# compatibility
ns_of=namespace_of


def name_of(s: str) -> str:
    """Last component of a fully qualified name"""
    p = s.rfind('.')
    return s[p+1:] if p >= 0 else s


def class_of(info: Union[str, dict]) -> str:
    if isinstance(info, str):
        return info
    for k in ["class", "class_name", "clazz", "method"]:
        if k in info:
            return info[k]
    raise ValueError("Missing mandatory key: ('class', 'class_name', 'clazz', 'method')")

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
