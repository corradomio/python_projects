from typing import Any, Optional, Union, cast
from path import Path as path


# ---------------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------------

def module_path(name=None) -> path:
    """
    Pyhon module of the current class
    """
    name = __name__ if name is None else name
    import sys
    this_path = path(sys.modules[name].__file__)
    return this_path.parent


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
    return f'{module}.{klass.__qualname__}'


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


def create_from(instance_info: str | dict, aliases=None) -> Any:
    """
    Create and instance of the object.

    instance_info: {
        "class": "qname|alias|type",
        "p1": <v1>, ...
    }

    aliases: {
        "alias": "qname|type"
    }
    
    """
    if instance_info is None:
        return None

    if aliases is None:
        aliases = {}
    if isinstance(instance_info, (str, type)):
        instance_info = {"class": instance_info}

    assert isinstance(instance_info, dict)
    assert isinstance(aliases, dict)

    qname: str = ""
    clazz_args = {}|instance_info

    # delete '#name' parameters
    for k in instance_info:
        if k.startswith("#"):
            del clazz_args[k]

    # delete one of the 'class' keywords
    for k in ["class", "class_name", "clazz", "method"]:
        if k in instance_info:
            qname = instance_info[k]
            if qname in aliases:
                qname = aliases[qname]
            del clazz_args[k]
            break

    assert isinstance(qname, str) and len(qname) > 0 or isinstance(qname, type), "Missing mandatory key: ('class', 'class_name', 'clazz', 'method')"
    if isinstance(qname, type):
        clazz = qname
    else:
        clazz = cast(type, import_from(qname))
    return clazz(**clazz_args)


def create_from_collection(collection: Union[None, list, dict]) -> Union[None, list, dict]:
    if collection is None:
        return None

    if isinstance(collection, dict):
        return {
            k: create_from(collection[k])
            for k in collection
        }
    elif isinstance(collection, list):
        return [
            create_from(e)
            for e in collection
        ]
    else:
        raise ValueError(f"Unsupported collection type {type(collection)}")

# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

def ns_of(s: str) -> str:
    """First component of a fully qualified name"""
    p = s.find('.')
    return s[:p]


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
