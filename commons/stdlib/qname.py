from typing import Any

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


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
