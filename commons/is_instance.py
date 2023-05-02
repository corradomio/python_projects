from typing import *
from typing import _UnionGenericAlias, _SpecialForm, _type_check, _remove_dups_flatten
from types import *
from collections import *

__all__ = [
    'is_instance'
]


#
#   Container   __contains__(
#   Iterable    __iter__
#   Hashable    __hash__
#   Sized       __len__
#   Callable    __call__
#   Collection  ??
#   Iterator    __iter__, __next__
#   Reversible  __reversed__
#   Awaitable   __await__
#   Coroutine   ??
#   AsyncIterable   __aiter__
#   AsyncIterator   __aiter__, __anext__
#   .

#     # Super-special typing primitives.
#     'Annotated',
#     'Any',
#     'Callable',
#     'ClassVar',
#     'Final',
#     'ForwardRef',
#     'Generic',
#     'Literal',
#     'Optional',
#     'Protocol',
#     'Tuple',
#     'Type',
#     'TypeVar',
#     'Union',
# 
#     # ABCs (from collections.abc).
#     'AbstractSet',  # collections.abc.Set.
#     'ByteString',
#     'Container',
#     'ContextManager',
#     'Hashable',
#     'ItemsView',
#     'Iterable',
#     'Iterator',
#     'KeysView',
#     'Mapping',
#     'MappingView',
#     'MutableMapping',
#     'MutableSequence',
#     'MutableSet',
#     'Sequence',
#     'Sized',
#     'ValuesView',
#     'Awaitable',
#     'AsyncIterator',
#     'AsyncIterable',
#     'Coroutine',
#     'Collection',
#     'AsyncGenerator',
#     'AsyncContextManager',
# 
#     # Structural checks, a.k.a. protocols.
#     'Reversible',
#     'SupportsAbs',
#     'SupportsBytes',
#     'SupportsComplex',
#     'SupportsFloat',
#     'SupportsIndex',
#     'SupportsInt',
#     'SupportsRound',
# 
#     # Concrete collection types.
#     'ChainMap',
#     'Counter',
#     'Deque',
#     'Dict',
#     'DefaultDict',
#     'List',
#     'OrderedDict',
#     'Set',
#     'FrozenSet',
#     'NamedTuple',  # Not really a type.
#     'TypedDict',  # Not really a type.
#     'Generator',
# 
#     # Other concrete types.
#     'BinaryIO',
#     'IO',
#     'Match',
#     'Pattern',
#     'TextIO',
# 
#     # One-off things.
#     'AnyStr',
#     'cast',
#     'final',
#     'get_args',
#     'get_origin',
#     'get_type_hints',
#     'NewType',
#     'no_type_check',
#     'no_type_check_decorator',
#     'NoReturn',
#     'overload',
#     'runtime_checkable',
#     'Text',
#     'TYPE_CHECKING',.


@_SpecialForm
def All(self, parameters):
    """Intersection type; All[X, Y] means X and Y.
    """
    if parameters == ():
        raise TypeError("Cannot take a All of no types.")
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    msg = "All[arg, ...]: each arg must be a type."
    parameters = tuple(_type_check(p, msg) for p in parameters)
    parameters = _remove_dups_flatten(parameters)
    if len(parameters) == 1:
        return parameters[0]
    uga = _UnionGenericAlias(self, parameters)
    uga._name = "All"
    return uga

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


class IsAny(IsInstance):
    def is_instance(self, obj) -> bool:
        return True


class IsNone(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        return obj is None


# ---------------------------------------------------------------------------

def _len(obj):
    try:
        return len(obj)
    except TypeError as e:
        return 0


class IsCollection(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = Collection

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, self.collection_type):
            return False
        # if not isinstance(obj, self.origin):
        #     return False

        if len(self.args) == 0:
            return True

        n = len(obj)
        if len(self.args) > 1 and len(self.args) != n:
            return False

        elif len(self.args) == 1:
            element_type = self.args[0]
            for item in obj:
                if not is_instance(item, element_type):
                    return False
        else:
            i = 0
            for item in obj:
                element_type = self.args[i]; i += 1;
                if not is_instance(item, element_type):
                    return False
        return True


class IsList(IsCollection):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = list


class IsTuple(IsCollection):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = tuple


class IsSet(IsCollection):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = set


class IsDeque(IsCollection):
    def __init__(self, tp):
        super().__init__(tp)
        self.collection_type = deque


# ---------------------------------------------------------------------------

class IsMapping(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        self.dictionary_type = Mapping

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, self.dictionary_type):
            return False

        key_type = self.args[0]
        value_type = self.args[1]

        for key in obj:
            value = obj[key]
            if not is_instance(key, key_type) or not is_instance(value, value_type):
                return False
        return True
# end

class IsDict(IsMapping):
    def __init__(self, tp):
        super().__init__(tp)
        self.dictionary_type = dict


class IsDefaultDict(IsMapping):
    def __init__(self, tp):
        super().__init__(tp)
        self.dictionary_type = defaultdict


# ---------------------------------------------------------------------------

class IsUnion(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        for a_type in self.args:
            if is_instance(obj, a_type):
                return True
        return False
# end

class IsAll(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        for a_type in self.args:
            if not is_instance(obj, a_type):
                return False
        return True
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


class IsNewType(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        return is_instance(obj, self.type.__supertype__)



# ---------------------------------------------------------------------------

class HasAttribute(IsInstance):
    def __init__(self, *attrs):
        self.attrs = attrs
        
    def __call__(self, *args, **kwargs):
        return self
        
    def is_instance(self, obj) -> bool:
        for attr in self.attrs:
            if not hasattr(obj, attr):
                return False
        return True
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

    'typing.None': IsNone,
    'typing.Union': IsUnion,
    'typing.All': IsAll,
    'typing.Any': IsAny,
    'typing.Optional': IsOptional,
    'typing.List': IsList,
    'typing.Tuple': IsTuple,
    'typing.Set': IsSet,
    'typing.Dict': IsDict,
    'typing.Deque': IsDeque,
    'typing.DefaultDict': IsDefaultDict,
    'typing.NewType': IsNewType,
    'typing.Collection': IsCollection,

    'typing.Container': HasAttribute('__contains__'),
    'typing.Iterable': HasAttribute('__iter__'),
    'typing.Hashable': HasAttribute('__hash__'),
    'typing.Sized': HasAttribute('__len__'),
    'typing.Callable': HasAttribute('__call__'),
    'typing.Iterator': HasAttribute('__iter__', '__next__'),
    'typing.Reversible': HasAttribute('__reversed__'),
    'typing.Awaitable': HasAttribute('__await__'),
    'typing.AsyncIterable': HasAttribute('__aiter__'),
    'typing.AsyncIterator': HasAttribute('__aiter__', '__anext__'),
}

def type_name(a_type: type) -> str:
    # if hasattr(a_type, '__origin__'):
    #     return str(a_type.__origin__)
    if hasattr(a_type, '__supertype__'):
        return f'typing.NewType'

    if hasattr(a_type, '_name'):
        name = a_type._name
    elif hasattr(a_type, '__name__'):
        name = a_type.__name__
    else:
        name = str(a_type)

    if name is None and hasattr(a_type, '__origin__'):
        t_name = str(a_type.__origin__)
    else:
        t_name = f'{a_type.__module__}.{name}'

    return t_name


def is_instance(obj, a_type: Union[type, Collection[type]]) -> bool:
    # if hasattr(a_type, '__supertype__'):
    #     return is_instance(obj, a_type.__supertype__)
    if isinstance(a_type, tuple):
        a_types: tuple[type] = a_type
        for a_type in a_types:
            if is_instance(obj, a_type):
                return True
        return False
    # end

    t_name = type_name(a_type)

    if t_name in IS_INSTANCE_OF:
        return IS_INSTANCE_OF[t_name](a_type).is_instance(obj)
    else:
        return isinstance(obj, a_type)
# end


isinstance = is_instance


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
