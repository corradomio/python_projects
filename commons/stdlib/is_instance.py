# ---------------------------------------------------------------------------
# Extension of 'isinstance' to support (partially) the Python type hints
# It is possible to use
#
#       is_instance(obj, Type)
#
# as alternative than
#
#       isinstance(obj, Type)
#
# where 'Type' can be specified as in Python type hints
#
#       https://docs.python.org/3/library/typing.html
#       https://peps.python.org/pep-0484/
#
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Parametrized types:
#   a type can be parametrized, as 'list[int]' if it defines
#
#       __class_getitem__(cls, item) -> GenericAlias
#
# ref: https://docs.python.org/3/reference/datamodel.html#object.__class_getitem__
# ---------------------------------------------------------------------------

#
#   _Final (typing)
#       _SpecialForm(_Final, _root=True) (typing)
#           _LiteralSpecialForm(_SpecialForm, _root=True) (typing)
#       ForwardRef(_Final, _root=True) (typing)
#       TypeVar(_Final, _Immutable, _root=True) (typing)
#       _BaseGenericAlias(_Final, _root=True) (typing)
#           _GenericAlias(_BaseGenericAlias, _root=True) (typing)
#               _CallableGenericAlias(_GenericAlias, _root=True) (typing)
#               _UnionGenericAlias(_GenericAlias, _root=True) (typing)
#               _LiteralGenericAlias(_GenericAlias, _root=True) (typing)
#               _AnnotatedAlias(_GenericAlias, _root=True) (typing)
#       _SpecialGenericAlias(_BaseGenericAlias, _root=True) (typing)
#               _CallableType(_SpecialGenericAlias, _root=True) (typing)
#               _TupleType(_SpecialGenericAlias, _root=True) (typing)
# .

__all__ = [
    'is_instance',
    'All',
    'Const',            # equivalent to 'Final'
    'Immutable',

    'IS_INSTANCE_OF',   # used for extensions
    'IsInstance'
]

__version__ = '1.0.1'

from typing import _type_check, _remove_dups_flatten
from typing import _GenericAlias, _UnionGenericAlias, _SpecialForm, _LiteralGenericAlias
from typing import _LiteralSpecialForm, _tp_cache, _flatten_literal_params, _deduplicate, _value_and_type_iter

# ---------------------------------------------------------------------------
# Typing types supported/unsupported
# ---------------------------------------------------------------------------
#
# Supported
# ---------
# Final
# Literal

#
# Unsupported
# -----------
# NoReturn
# ClassVar
# TypeAlias
# Concatenate
# TypeGuard
# ForwardRef
# TypeVar
# .


# ---------------------------------------------------------------------------
# from collections.abc, and available in typing
# ---------------------------------------------------------------------------
from collections import *
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
# .


# ---------------------------------------------------------------------------
# from types
# ---------------------------------------------------------------------------
from types import *
#   FunctionType
#   LambdaType
#   CodeType
#   MappingProxyType
#   SimpleNamespace
#   CellType
#   GeneratorType
#   CoroutineType
#   AsyncGeneratorType
#   MethodType
#   BuiltinMethodType
#   WrapperDescriptorType
#   MethodWrapperType
#   MethodDescriptorType
#   ClassMethodDescriptorType
#   ModuleType
#   GetSetDescriptorType
#   MemberDescriptorType
# .

# ---------------------------------------------------------------------------
# from typing
# ---------------------------------------------------------------------------
from typing import *
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
#     'TYPE_CHECKING',
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# from 'collections'
# ---------------------------------------------------------------------------
from collections import *
#     'ChainMap',
#     'Counter',
#     'OrderedDict',
#     'UserDict',
#     'UserList',
#     'UserString',
#     'defaultdict',
#     'deque',
#     'namedtuple',
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Special forms
# ---------------------------------------------------------------------------

@_LiteralSpecialForm
@_tp_cache(typed=True)
def Literals(self, *parameters):
    parameters = _flatten_literal_params(parameters[0])

    try:
        parameters = tuple(p for p, _ in _deduplicate(list(_value_and_type_iter(parameters))))
    except TypeError:  # unhashable parameters
        pass

    return _LiteralGenericAlias(self, parameters)


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


@_SpecialForm
def Immutable(self, parameters=()):
    """Special typing construct to indicate  immutable to type checkers.

        A immutable object must be composed by only immutable objects
    """
    item = _type_check(parameters, f'{self} accepts only single type.')
    return _GenericAlias(self, (item,))


@_SpecialForm
def Const(self, parameters):
    """Special typing construct to indicate const names to type checkers.

        A const name cannot be re-assigned.
        For example:

          MAX_SIZE: Const[int] = 9000
          MAX_SIZE += 1  # Error reported by type checker

        """
    item = _type_check(parameters, f'{self} accepts only single type.')
    return _GenericAlias(self, (item,))


# ---------------------------------------------------------------------------
# IsInstance (root)
#   IsAny
#   IsAll
#   IsNone
#   IsCollection
#       IsList
#       IsTuple
#       IsSet
#       IsDeque
#       ...
#   IsMapping
#       IsDict
#       ...
#   ...
#   HasAttribute
# ---------------------------------------------------------------------------

class IsInstance:
    def __init__(self, tp):
        self.type = tp
        self.origin = get_origin(tp)
        self.args = get_args(tp)
        self.n = len(self.args)

    def is_instance(self, obj) -> bool:
        ...


# ---------------------------------------------------------------------------

class IsAny(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        return True


class IsNone(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        return obj is None


class IsLiteral(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        self.targs = [type(arg) for arg in self.args]
        self.nargs = len(self.args)
        self.values = tp.__args__

    def is_instance(self, obj) -> bool:
        # Problem:  1 == True and 0 == False
        # BUT, for type consistency, this is not a good idea
        # both values must have the same type
        # tobj = type(obj)
        # for i in range(self.nargs):
        #     if obj == self.args[i] and tobj == self.targs[i]:
        #         return True
        # return False
        return obj in self.values


class IsLiterals(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        self.targs = [type(arg) for arg in self.args]
        self.nargs = len(self.args)
        self.values = tp.__args__

    def is_instance(self, obj) -> bool:
        # Problem:  1 == True and 0 == False
        # BUT, for type consistency, this is not a good idea
        # both values must have the same type
        # tobj = type(obj)
        # for i in range(self.nargs):
        #     if obj == self.args[i] and tobj == self.targs[i]:
        #         return True
        # return False
        return obj in self.values


# ---------------------------------------------------------------------------

def _len(obj):
    try:
        return len(obj)
    except TypeError as e:
        return 0


class IsCollection(IsInstance):
    # supported:  collection[T], collection[T1,T2,...]
    def __init__(self, tp, collection_type=None):
        super().__init__(tp)
        if collection_type is None:
            collection_type = (list, tuple, namedtuple, set, frozenset, deque)
        self.collection_type = collection_type

    def is_instance(self, obj) -> bool:
        # Problem: how to check a collection with arbitrary length vs a collection with a specified length?
        # A possibility is:
        #   collection[T, ...]
        #       collection composed by 0+ elements of type T
        #   collection[T1] | collection[T1,T2] etc
        #       collection composed exactly of n elements of the selected types (in the specified order)

        if not isinstance(obj, self.collection_type):
            return False
        # if not isinstance(obj, self.origin):
        #     return False

        # is_instance(x, collection)
        if len(self.args) == 0:
            return True

        # is_instance(x, collection[T1, T2, ...])
        #   check if x contains 0+ items of type T
        if len(self.args) > 0 and self.args[-1] == ...:
            ntypes = len(self.args)-1
            for i, item in enumerate(obj):
                element_type = self.args[i] if i < ntypes else self.args[ntypes-1]
                if not is_instance(item, element_type):
                    return False
            return True

        # if collection[T]
        #   check if x contains 0+ items of type T
        # if collection[T1,...Tn]
        #   check if x contains exactly n items of type Ti

        # length of collection is different than the number of parameters
        n = len(obj)
        if len(self.args) > 1 and len(self.args) != n:
            return False

        # a collection of a single element is not very useful.
        # the, if specified 'collection[T]', it will check for
        # collection with 0+ elements of type T
        elif len(self.args) == 1:
            element_type = self.args[0]
            for item in obj:
                if not is_instance(item, element_type):
                    return False
        else:
            i = 0
            for item in obj:
                element_type = self.args[i]; i += 1
                if not is_instance(item, element_type):
                    return False
        return True


class IsList(IsCollection):
    # supported:  list[T], list[T1,T2,...]
    def __init__(self, tp):
        super().__init__(tp, list)


class IsTuple(IsCollection):
    # supported:  tuple[T], tuple[T1,T2,...]
    def __init__(self, tp):
        super().__init__(tp, tuple)


class IsNamedTuple(IsCollection):
    def __init__(self, tp):
        super().__init__(tp, namedtuple)


class IsSet(IsCollection):
    def __init__(self, tp):
        super().__init__(tp, set)


class IsFrozenSet(IsCollection):
    def __init__(self, tp):
        super().__init__(tp, frozenset)


class IsDeque(IsCollection):
    def __init__(self, tp):
        super().__init__(tp, deque)


# ---------------------------------------------------------------------------

class IsMapping(IsInstance):
    def __init__(self, tp, dictionary_type=Mapping):
        super().__init__(tp)
        self.dictionary_type = dictionary_type

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, self.dictionary_type):
            return False
        
        # no type parameters passed: true for ALL types
        if len(self.args) == 0:
            return True

        key_type = self.args[0]
        value_type = self.args[1]

        for key in obj:
            value = obj[key]
            if not is_instance(key, key_type) or not is_instance(value, value_type):
                return False
        return True
    
    
class IsDict(IsMapping):
    def __init__(self, tp):
        super().__init__(tp, dict)


class IsDefaultDict(IsMapping):
    def __init__(self, tp):
        super().__init__(tp, defaultdict)


class IsOrderedDict(IsMapping):
    def __init__(self, tp):
        super().__init__(tp, OrderedDict)


# ---------------------------------------------------------------------------

class IsUnion(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        for a_type in self.args:
            if is_instance(obj, a_type):
                return True
        return False
    

class IsOptional(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if obj is None:
            return True
        for a_type in self.args:
            if is_instance(obj, a_type):
                return True
        return False


class IsNewType(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        return is_instance(obj, self.type.__supertype__)


# ---------------------------------------------------------------------------

class IsAll(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        for a_type in self.args:
            if not is_instance(obj, a_type):
                return False
        return True


class IsImmutable(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        
    def is_instance(self, obj) -> bool:
        o_type = type(obj)
        if o_type in [None, int, float, bool, str, bytes, float]:
            return True
        if o_type not in [tuple, frozenset]:
            return False
        for e in obj:
            if not self.is_instance(e):
                return False
        return True
    
    
class IsConst(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if len(self.args) == 0:
            return True
        else:
            return is_instance(obj, self.args[0])


# ---------------------------------------------------------------------------

class HasAttribute(IsInstance):

    def __init__(self, *attrs):
        super().__init__(type(None))
        self.attrs = attrs
        
    def __call__(self, *args, **kwargs):
        return self
        
    def is_instance(self, obj) -> bool:
        for attr in self.attrs:
            if not hasattr(obj, attr):
                return False
            av = getattr(obj, attr)
            if av is None:
                return False
        return True


class HasMethods(IsInstance):

    def __init__(self, methods):
        super().__init__(type(None))
        self.methods = methods

    def is_instance(self, obj) -> bool:
        for method in self.methods:
            if not hasattr(obj, method):
                return False
        return True


# ---------------------------------------------------------------------------

class IsLiteralExtend(IsInstance):
    def __init__(self, value):
        super().__init__(type(None))
        self.value = value

    def is_instance(self, obj) -> bool:
        return obj == self.value


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

IS_INSTANCE_OF = {
    'builtins.NoneType': IsNone,
    'builtins.list': IsList,
    'builtins.tuple': IsTuple,
    'builtins.set': IsSet,
    'builtins.dict': IsDict,
    'builtins.frozenset': IsFrozenSet,

    'collections.deque': IsDeque,
    'collections.defaultdict': IsDefaultDict,
    'collections.namedtuple': IsNamedTuple,
    'collections.OrderedDict': IsOrderedDict,

    'typing.None': IsNone,
    'typing.Union': IsUnion,
    'typing.Any': IsAny,
    'typing.Optional': IsOptional,
    'typing.Literal': IsLiteral,

    'typing.List': IsList,
    'typing.Tuple': IsTuple,
    'typing.NamedTuple': IsNamedTuple,
    'typing.Set': IsSet,
    'typing.FrozenSet': IsFrozenSet,
    'typing.Deque': IsDeque,

    'typing.Dict': IsDict,
    'typing.DefaultDict': IsDefaultDict,
    'typing.NewType': IsNewType,
    'typing.Collection': IsCollection,
    'typing.Mapping': IsMapping,

    'typing.All': IsAll,
    'typing.Immutable': IsImmutable,
    'typing.Const': IsConst,

    'types.UnionType': IsUnion,

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

    'extend.Literal': IsLiteralExtend,
}


def type_name(a_type: type) -> str:
    # if hasattr(a_type, '__origin__'):
    #     return str(a_type.__origin__)
    if a_type is None:
        return 'builtins.NoneType'
    elif isinstance(a_type, (int, str)):
        return 'extend.Literal'
    elif isinstance(a_type, _LiteralGenericAlias):
        return 'typing.Literal'
    elif isinstance(a_type, _GenericAlias):
        a_type = a_type.__origin__
    elif hasattr(a_type, '__supertype__'):
        return f'typing.NewType'

    if hasattr(a_type, '_name'):
        name = a_type._name
    elif hasattr(a_type, '__name__'):
        name = a_type.__name__
    elif hasattr(a_type, '__class__'):
        # name = f"{a_type.__class__.__module__}.{a_type.__class__.__name__}"
        name = a_type.__class__.__name__
    else:
        name = str(a_type)

    if name is None and hasattr(a_type, '__origin__'):
        t_name = str(a_type.__origin__)
    else:
        t_name = f'{a_type.__module__}.{name}'

    return t_name


def is_instance(obj, a_type, msg=None) -> bool:

    # if hasattr(a_type, '__supertype__'):
    #     return is_instance(obj, a_type.__supertype__)

    # is_instance(obj, (T1, T2, ...))
    if isinstance(a_type, tuple):
        a_types: tuple[type] = a_type
        for a_type in a_types:
            if is_instance(obj, a_type):
                return True
        return False

    # is_instance(obj, Union[...])
    t_name = type_name(a_type)

    if t_name in IS_INSTANCE_OF:
        valid = IS_INSTANCE_OF[t_name](a_type).is_instance(obj)
    elif isinstance(a_type, IsInstance):
        valid = a_type.is_instance(obj)
    elif not isinstance(a_type, type) and callable(a_type):
        return a_type(obj)
    else:
        valid = isinstance(obj, a_type)

    if not valid and msg is not None:
        raise AssertionError(msg)
    return valid


def has_methods(obj_or_methods: Union[object, list[str]], methods: list[str] = None, msg=None) \
        -> Union[bool, HasMethods]:
    if isinstance(obj_or_methods, (list, tuple)):
        methods: list[str] = list(obj_or_methods)
        return HasMethods(methods)
    missing = []
    obj = obj_or_methods
    for m in methods:
        if not hasattr(obj, m):
            missing.append(m)
    valid = len(missing) == 0
    if not valid and msg is not None:
        raise AssertionError(msg + " " + missing)
    return valid


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
