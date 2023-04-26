from typing import Any

NoneType = type(None)


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
# end


def qualified_name(clazz: type) -> str:
    return f'{clazz.__module__}.{clazz.__name__}'

