from typing import *
from types import *

__all__ = [
    'is_instance'
]


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class IsInstance:
    def __init__(self, tp):
        self.type = tp
        self.origin = get_origin(tp)
        self.args = get_args(tp)
        self.n = len(self.args)

    def is_instance(self, obj) -> bool:
        ...
# end


class IsAny(IsInstance):
    def is_instance(self, obj) -> bool:
        return True


class IsNone(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        return obj is None
# end


class IsListOrTuple(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = None

    def is_instance(self, obj) -> bool:
        collection_type = self.collection_type
        if not isinstance(obj, collection_type):
            return False

        if len(self.args) == 0:
            return True

        n = len(obj)
        if len(self.args) > 1 and len(self.args) != n:
            return False
        elif len(self.args) == 1:
            element_type = self.args[0]
            for i in range(n):
                if not is_instance(obj[i], element_type):
                    return False
        else:
            for i in range(n):
                element_type = self.args[i]
                if not is_instance(obj[i], element_type):
                    return False
        return True
# end


class IsList(IsListOrTuple):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = list


class IsTuple(IsListOrTuple):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = tuple


class IsSet(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, set):
            return False
        if len(self.args) == 0:
            return True

        element_type = self.args[0]
        for e in obj:
            if not is_instance(e, element_type):
                return False
        return True
# end


class IsDict(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, dict):
            return False

        key_type = self.args[0]
        value_type = self.args[1]

        for key in obj:
            value = obj[key]
            if not is_instance(key, key_type) or not is_instance(value, value_type):
                return False
        return True
# end


class IsUnion(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        for i in range(self.n):
            if is_instance(obj, self.args[i]):
                return True
        return False
# end


class IsOptional(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if obj is None:
            return True
        else:
            return is_instance(obj, self.args[0])
# end


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

IS_INSTANCE_OF = {
    'builtins.NoneType': IsNone,
    'builtins.list': IsList,
    'builtins.tuple': IsTuple,
    'builtins.set': IsSet,
    'builtins.dict': IsDict,

    'typing.Union': IsUnion,
    'typing.Any': IsAny,
    'typing.Optional': IsOptional,
    'typing.List': IsList,
    'typing.Tuple': IsTuple,
    'typing.Set': IsSet,
    'typing.Dict': IsDict,
}

def type_name(a_type: type) -> str:
    if hasattr(a_type, "_name"):
        return f'{a_type.__module__}.{a_type._name}'
    else:
        return f'{a_type.__module__}.{a_type.__name__}'


def is_instance(obj, a_type: type) -> bool:
    t_name = type_name(a_type)

    if t_name in IS_INSTANCE_OF:
        return IS_INSTANCE_OF[t_name](a_type).is_instance(obj)
    else:
        return isinstance(obj, a_type)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
