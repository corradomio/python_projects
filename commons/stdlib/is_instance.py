# ---------------------------------------------------------------------------
# Python Data Model
#
#   object has identity and type
#   id(obj)
#   type(obj)
#
#   None: type(None)
#   NotImplemented: type(NotImplemented)
#   .../Ellipsis: type(Ellipsis)
#   Number
#       Integral: int, bool
#       Real: float
#       Complex: complex
#   Sequences
#       immutable: bytes, str, tuple
#       mutable: list, bytearray
#   Sets: set, frozenset
#   Mapping: dict
#   Callable:
#       -- functions --
#       readonly attrs: f.[__builtins__, __globals__, __closure__]
#       writable attrs: f.[__doc__, __name__, __qualname__, __module__, __defauls__, __code__, __dict__,
#                          __annotations__, __annotate__, __kwdefaults__, __type_params__]
#       -- methods --
#       readonly: m.[__self__, __func__, __doc__, __name__, __module__]
#
#   Class/type
#       attrs: t.[__name__, __qualname__, __module__, __dict__, __bases__, __base__, __doc__, __annotations__,
#                 __annotate__, __type_params__, __static_attributes__, __firstlineno__, __mro__]
#       t.mro(), t.__subclasses__
#   Instance
#       attrs: obj.[__class__, __dict__ | __slots__].

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
# Extended annotations:
#
#   Immutable       immutable object composed by immutable sub-objects
#   Intersection[T1,T2,...]  the object must be of ALL types
#

# ---------------------------------------------------------------------------
# Parametrized types:
#   a type can be parametrized, as 'list[int]' if it defines
#
#       __class_getitem__(cls, item) -> GenericAlias
#
# ref: https://docs.python.org/3/reference/datamodel.html#object.__class_getitem__

# ---------------------------------------------------------------------------
# Interresting imports
#
# abc
# typing
# types
# collections
# collections.abc
#

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
#               _ConcatenateGenericAlias(_GenericAlias, _root=True)
#               _UnpackGenericAlias(_GenericAlias, _root=True)
#       _SpecialGenericAlias(_BaseGenericAlias, _root=True) (typing)
#               _CallableType(_SpecialGenericAlias, _root=True) (typing)
#               _TupleType(_SpecialGenericAlias, _root=True) (typing)
#               _DeprecatedGenericAlias(_SpecialGenericAlias, _root=True)
#
#   _NotIterable (typing)
#       _SpecialForm(_Final, _NotIterable, _root=True) (typing)
#           _LiteralSpecialForm(_SpecialForm, _root=True) (typing)
#       _SpecialGenericAlias(_NotIterable, _BaseGenericAlias, _root=True) (typing)
#           _DeprecatedGenericAlias(_SpecialGenericAlias, _root=True) (typing)
#           _CallableType(_SpecialGenericAlias, _root=True) (typing)
#           _TupleType(_SpecialGenericAlias, _root=True) (typing)
#       _CallableGenericAlias(_NotIterable, _GenericAlias, _root=True) (typing)
#       _UnionGenericAlias(_NotIterable, _GenericAlias, _root=True) (typing)
#       _AnnotatedAlias(_NotIterable, _GenericAlias, _root=True) (typing).

__all__ = [
    'is_instance',

    'Intersection',     # Intersection[T1,...]
    'Immutable',        # Immutable | Immutable[T]
    "Literals",         # Literals[[l1, ...]]   as alternative to Literal[l1, ...]
    "Supertype",        # Supertype[T1, ...]
    "Not",              # Not[T]
    "InRange",          # InRange[L,U]

    # support for extensions
    'IS_INSTANCE_OF',   # dictionary
    'IsInstance'        # base class
]

__version__ = '1.0.2'

import typing
from abc import ABCMeta
from typing import _type_check, _tp_cache
from typing import _GenericAlias, _SpecialGenericAlias
from typing import _UnionGenericAlias, _SpecialForm, _LiteralGenericAlias

try:
    # Python 3.12
    from typing import _LiteralSpecialForm
except:
    # Python 3.14
    from typing import _TypedCacheSpecialForm
    _LiteralSpecialForm = _TypedCacheSpecialForm

try:
    # Python 3.12
    from typing import TypeAliasType
except:
    # Python 3.10
    from typing import TypeAlias
    TypeAliasType = TypeAlias

try:
    # Python 3.13
    from typing import is_protocol, get_protocol_members
except:
    # Python 3.12
    from typing import Protocol
    def is_protocol(tp):
        try:
            return Protocol in tp.__mro__
        except:
            return False

    def get_protocol_members(tp):
        protocol_members = set(Protocol.__dict__.keys())
        tp_members = set(tp.__dict__.keys())
        unused_members = {'__annotations__', '__init__', '__subclasshook__'}
        useful_members = tp_members - protocol_members - unused_members
        return frozenset(useful_members)


def get_metaclass_members(tp, metaclass=ABCMeta):
    try:
        metaclass_members = set(metaclass.__dict__.keys())
        tp_members = set(tp.__dict__.keys())
        unused_members = {
            '__annotations__', '__init__', '__subclasshook__',                      # Protocol
            '__abstractmethods__', '__slots__', '__subclasshook__', '_abc_impl',    # ABCMeta
            '__class_getitem__'
        }
        useful_members = tp_members - metaclass_members - unused_members
        return frozenset(useful_members)
    except:
        return frozenset()


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
# from collections.abc, and available in typing
# ---------------------------------------------------------------------------
from collections.abc import *
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
# Utilities
# ---------------------------------------------------------------------------

def has_metaclass(tp, metaclass=ABCMeta):
    if isinstance(tp, _SpecialGenericAlias):
        tp = get_origin(tp)
    return issubclass(type(tp), metaclass)


def get_name(tp):
    if hasattr(tp, '_name'):
        name = tp._name
    elif hasattr(tp, '__name__'):
        name = tp.__name__
    elif hasattr(tp, '__class__'):
        # name = f"{a_type.__class__.__module__}.{a_type.__class__.__name__}"
        name = tp.__class__.__name__
    else:
        name = str(tp)

    if name is None and hasattr(tp, '__origin__'):
        name = str(tp.__origin__)
    else:
        name = f'{tp.__module__}.{name}'

    return name


def set_name(obj, name):
    if hasattr(obj, '_name'):
        obj._name = name
    elif hasattr(obj, '__name__'):
        setattr(obj, '__name__', name)
    else:
        setattr(obj, '_name', name)


# ---------------------------------------------------------------------------
# Special forms
# ---------------------------------------------------------------------------

@_LiteralSpecialForm
@_tp_cache(typed=True)
def Literals(self, *parameters):
    # parameters = _flatten_literal_params(parameters[0])

    try:
        # parameters = tuple(p for p, _ in _deduplicate(list(_value_and_type_iter(parameters))))
        parameters = tuple(p for p in parameters)
    except TypeError:  # unhashable parameters
        pass

    return _LiteralGenericAlias(self, parameters)


@_SpecialForm
def Intersection(self, parameters):
    """
    Intersection type; Intersection[X, Y, ...] means X and Y.
    Possible syntax:

        X & Y & ...

    as Union[X, Y, ...] is equivalent to

        X | Y | ...

    """
    if parameters == ():
        raise TypeError("Cannot take a Intersection of no types.")
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    msg = "Intersection[T, ...]: each T must be a type."
    parameters = tuple(_type_check(p, msg) for p in parameters)
    uga = _GenericAlias(self, parameters, name="Intersection")
    return uga


@_SpecialForm
def Immutable(self, parameters=()):
    """
    Immutable | Immutable[T]
    Special typing construct to indicate  immutable to type checkers.
    A immutable object must be composed by only immutable objects
    """
    item = _type_check(parameters, f'{self} accepts only single type.')
    return _GenericAlias(self, (item,))


@_SpecialForm
def Supertype(self, parameters):
    """
    Supertype[T1,...]
    An object is of type a supertype of some specified type.
    """
    msg = "Supertype[T, ...]: each T must be a type."
    if not isinstance(parameters, tuple):
        parameters = (parameters,)
    parameters = tuple(_type_check(p, msg) for p in parameters)
    return _GenericAlias(self, parameters)



@_SpecialForm
def Not(self, parameter):
    """
    Not[T]
    An object is NOT of the specified type
    """
    msg = "Not[T]: T must be a type."
    parameter = _type_check(parameter, msg)
    return _GenericAlias(self, parameter)


@_SpecialForm
def InRange(self, *parameters):
    """
    InRange[L,U]
    L and U must be float or integer
    """
    msg = "InRange[L,U]: L and U must be float or integer."
    return _GenericAlias(self, parameters)


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
        self.nargs = len(self.args)

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

    def is_instance(self, obj) -> bool:
        # Problem:  1 == True and 0 == False
        # BUT, for type consistency, this is not a good idea
        # both values must have the same type
        return obj in self.args


class IsType(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        assert self.nargs == 1

    def is_instance(self, obj) -> bool:
        return is_instance(obj, self.args[0])


class IsSupertype(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        ot = type(obj)
        for st in self.args:
            if issubclass(st, ot):
                return True
        return False


# ---------------------------------------------------------------------------

class IsCollection(IsInstance):
    # supported:  collection[T], collection[T1,T2,...]
    def __init__(self, tp, collection_type=None):
        super().__init__(tp)
        if collection_type is None:
            collection_type = (list, tuple, set, frozenset, deque)
            # collection_type += (namedtuple)
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
        if self.nargs == 0:
            return True

        # is_instance(x, collection[T1, T2, ...])
        #   check if x contains 0+ items of type T
        if self.nargs > 0 and self.args[-1] == ...:
            ntypes = self.nargs-1
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
        if self.nargs > 1 and self.nargs != n:
            return False

        # a collection of a single element is not very useful.
        # the, if specified 'collection[T]', it will check for
        # collection with 0+ elements of type T
        elif self.nargs == 1:
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

class IsSequence(IsCollection):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, (list, tuple, str)):
            return False
        else:
            return super().is_instance(obj)


# ---------------------------------------------------------------------------

class IsMapping(IsInstance):
    def __init__(self, tp, dictionary_type=Mapping):
        super().__init__(tp)
        self.dictionary_type = dictionary_type

    def is_instance(self, obj) -> bool:
        if not isinstance(obj, self.dictionary_type):
            return False
        
        # no type parameters passed: true for ALL types
        if self.nargs == 0:
            return True

        key_type = self.args[0]
        value_type = self.args[1]

        for key in obj:
            value = obj[key]
            if not is_instance(key, key_type):
                return False
            if not is_instance(value, value_type):
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


class IsOfType(IsInstance):
    def __init__(self, tp_list):
        super().__init__(type(None))
        self.args = tp_list
        self.nargs = len(tp_list)

    def is_instance(self, obj) -> bool:
        for a_type in self.args:
            if is_instance(obj, a_type):
                return True
        return False


class IsIntersection(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        for a_type in self.args:
            if not is_instance(obj, a_type):
                return False
        return True


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
        if self.nargs == 0:
            return True
        else:
            return is_instance(obj, self.args[0])


class IsFinal(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)

    def is_instance(self, obj) -> bool:
        if self.nargs == 0:
            return True
        else:
            return is_instance(obj, self.args[0])


# ---------------------------------------------------------------------------

class HasAttribute(IsInstance):

    def __init__(self, *attrs):
        super().__init__(type(None))
        self.args = attrs
        self.nargs = len(attrs)
        
    def __call__(self, *args, **kwargs):
        return self
        
    def is_instance(self, obj) -> bool:
        for attr in self.args:
            if not hasattr(obj, attr):
                return False
            av = getattr(obj, attr)
            if av is None:
                return False
        return True


class HasMethod(IsInstance):

    def __init__(self, methods):
        super().__init__(type(None))
        self.args = methods
        self.nargs = len(methods)

    def is_instance(self, obj) -> bool:
        return self.validate(obj, None)

    def validate(self, obj, msg):
        missing = []
        for m in self.args:
            if not hasattr(obj, m):
                missing.append(m)
        if len(missing) > 0 and msg is not None:
            raise AssertionError(msg + " " + missing)
        else:
            return len(missing) == 0


class IsProtocol(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        self.args = get_protocol_members(self.type)
        self.nargs = len(self.args)

    def is_instance(self, obj) -> bool:
        for method in self.args:
            if not hasattr(obj, method):
                return False
        return True


class IsProtocolMetaclass(IsInstance):
    def __init__(self, tp):
        super().__init__(get_origin(tp))
        self.args = get_metaclass_members(self.type)
        self.nargs = len(self.args)

    def is_instance(self, obj) -> bool:
        for method in self.args:
            if not hasattr(obj, method):
                return False
            if getattr(obj, method) is None:
                return False
        return True


# ---------------------------------------------------------------------------

class IsLiteralExtend(IsInstance):
    def __init__(self, value):
        super().__init__(type(None))
        self.value = value

    def is_instance(self, obj) -> bool:
        return obj == self.value


class IsNot(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        assert self.nargs == 1

    def is_instance(self, obj) -> bool:
        return not is_instance(obj, self.args[0])


class IsInRange(IsInstance):
    def __init__(self, tp):
        super().__init__(tp)
        assert self.nargs == 1

    def is_instance(self, obj) -> bool:
        lower, upper = self.args[0]
        return isinstance(obj, (int,float)) and (lower <= obj <= upper)


# ---------------------------------------------------------------------------
# is_instance
# ---------------------------------------------------------------------------

IS_INSTANCE_OF = {
    'builtins.NoneType': IsNone,
    'builtins.list': IsList,
    'builtins.tuple': IsTuple,
    'builtins.set': IsSet,
    'builtins.dict': IsDict,
    'builtins.frozenset': IsFrozenSet,
    'builtins.type': IsType,

    'collections.deque': IsDeque,
    'collections.defaultdict': IsDefaultDict,
    'collections.namedtuple': IsNamedTuple,
    'collections.OrderedDict': IsOrderedDict,

    'collections.abc.Mapping': IsMapping,

    'typing.None': IsNone,
    'typing.Union': IsUnion,
    'typing.Any': IsAny,
    'typing.Optional': IsOptional,
    'typing.Literal': IsLiteral,
    'typing.Type': IsType,
    'typing.Supertype': IsSupertype,

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

    'typing.Intersection': IsIntersection,
    'typing.Immutable': IsImmutable,
    'typing.Const': IsConst,
    'typing.Final': IsFinal,

    'types.UnionType': IsUnion,

    'collections.abc.Collection': IsCollection,
    # 'typing.Sequence': IsSequence,

    # 'typing.Container': HasAttribute('__contains__'),
    # 'typing.Iterable': HasAttribute('__iter__'),
    # 'typing.Hashable': HasAttribute('__hash__'),
    # 'typing.Sized': HasAttribute('__len__'),
    # 'typing.Callable': HasAttribute('__call__'),
    # 'typing.Iterator': HasAttribute('__iter__', '__next__'),
    # 'typing.Reversible': HasAttribute('__reversed__'),
    'typing.Awaitable': HasAttribute('__await__'),
    'typing.AsyncIterable': HasAttribute('__aiter__'),
    'typing.AsyncIterator': HasAttribute('__aiter__', '__anext__'),

    'typing.LiteralExtended': IsLiteralExtend,
    'typing.Not': IsNot,
    'typing.InRange': IsInRange,
}


def type_name(a_type: type) -> str:
    # if hasattr(a_type, '__origin__'):
    #     return str(a_type.__origin__)
    if a_type is None:
        return 'builtins.NoneType'
    elif isinstance(a_type, (int, str)):
        return 'typing.LiteralExtended'
    elif isinstance(a_type, _LiteralGenericAlias):
        return 'typing.Literal'
    elif isinstance(a_type, _GenericAlias):
        # a_type = a_type.__origin__
        a_type = get_origin(a_type)
    elif hasattr(a_type, '__supertype__'):
        return f'typing.NewType'

    t_name = get_name(a_type)

    return t_name


def is_instance(obj, a_type, msg=None) -> bool:

    # is_instance(obj, (T1, T2, ...))
    if isinstance(a_type, (tuple, list)):
        return IsOfType(a_type).is_instance(obj)

    if is_protocol(a_type):
        return IsProtocol(a_type).is_instance(obj)

    if has_metaclass(a_type):
        # t_name = get_name(a_type)
        # if t_name in IS_INSTANCE_OF:
        #     print(f"WARN: {t_name} is already a (ABCMeta)Protocol")
        return IsProtocolMetaclass(a_type).is_instance(obj)

    try:
        # Python 3.14
        alias_type = None
        alias_params = ()
        if isinstance(a_type, TypeAliasType):
            alias_type = a_type.__value__
            alias_params = a_type.__type_params__
            if len(alias_params) == 0:
                a_type = alias_type
            else:
                a_type = alias_type[alias_params]
    except:
        pass

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
        -> Union[bool, HasMethod]:
    """
    Check if the obj has the specified list of methods.
    If used as

        has_methods(['fit', 'predict'])

    return an instance of the class HasMethods(methods).
    If used as

        has_methods(obj, ['fit', 'predict'])

    it returns True or False.
    If used as

        has_methods(obj, ['fit', 'predict'], "Invalid object)

    if raises an

    :param obj_or_methods: object instance or a list of methods
    :param methods: None or list of methods
    :param msg: None or message to use in AssertionError
    :return:
    """
    if isinstance(obj_or_methods, (list, tuple, frozenset, set)):
        methods: list[str] = list(obj_or_methods)
        return HasMethod(methods)
    else:
        return HasMethod(methods).validate(obj_or_methods, msg)


def cast(a_type, obj) -> object:
    assert is_instance(a_type, obj)
    return obj


# ---------------------------------------------------------------------------
# Optional module: numpy
# ---------------------------------------------------------------------------
# np.ndarray
# np.ndarray[Any]
# np.ndarray[tuple[int,int]]
# np.ndarray[Any, np.dtype[np.float[16]]
# np.ndarray[tuple[int,int], np.dtype[np.float[16]]
# np.ndarray[(1,2), ...]
#
# generic(_ArrayOrScalarCommon, Generic[_ItemT_co]) (numpy stub)
#     rational(__numpy.generic) (numpy._core._rational_tests)
#     number(generic[_NumberItemT_co], Generic[_NBit, _NumberItemT_co]) (numpy stub)
#         integer(_IntegralMixin, _RoundMixin, number[_NBit, int]) (numpy stub)
#             signedinteger(__numpy.integer) (numpy._core._multiarray_umath)
#             unsignedinteger(__numpy.integer) (numpy._core._multiarray_umath)
#             signedinteger(integer[_NBit]) (numpy stub)
#             unsignedinteger(integer[_NBit1]) (numpy stub)
#         inexact(number[_NBit, _InexactItemT_co], Generic[_NBit, _InexactItemT_co]) (numpy stub)
#             complexfloating(__numpy.inexact) (numpy._core._multiarray_umath)
#             floating(__numpy.inexact) (numpy._core._multiarray_umath)
#             floating(_RealMixin, _RoundMixin, inexact[_NBit1, float]) (numpy stub)
#                 float64(floating[_64Bit], float) (numpy stub)
#             complexfloating(inexact[_NBit1, complex], Generic[_NBit1, _NBit2]) (numpy stub)
#                 complex128(complexfloating[_64Bit, _64Bit], complex) (numpy stub)
#     bool(generic[_BoolItemT_co], Generic[_BoolItemT_co]) (numpy stub)
#     object_(_RealMixin, generic) (numpy stub)
#     timedelta64(_IntegralMixin, generic[_TD64ItemT_co], Generic[_TD64ItemT_co]) (numpy stub)
#     datetime64(_RealMixin, generic[_DT64ItemT_co], Generic[_DT64ItemT_co]) (numpy stub)
#     flexible(_RealMixin, generic[_FlexibleItemT_co], Generic[_FlexibleItemT_co]) (numpy stub)
#         character(__numpy.flexible) (numpy._core._multiarray_umath)
#         void(flexible[bytes | tuple[Any, ...]]) (numpy stub)
#             record(nt.void) (numpy._core.records)
#             record(np.void) (numpy._core.records stub)
#         character(flexible[_CharacterItemT_co], Generic[_CharacterItemT_co]) (numpy stub)
#             bytes_(character[bytes], bytes) (numpy stub)
#             str_(character[str], str) (numpy stub)

try:
    import numpy as np

    NP_GENERIC = np.generic

    PY_TYPES = (bool, float, int, complex)

    NP_TYPES = (
        np.bool, np.integer, np.inexact, np.floating,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
        np.complex64, np.complex128
    )


    class IsNumpyArray(IsInstance):
        def __init__(self, tp):
            super().__init__(tp)

        def is_instance(self, obj) -> bool:
            if not isinstance(obj, np.ndarray):
                return False

            arry: np.ndarray = obj
            n_args = len(self.args)
            if n_args == 0:
                return True

            # shape
            if n_args >=1:
                shape = self.args[0]
                # Any
                if shape == Any:
                    pass
                # (1,2)
                elif isinstance(shape, tuple|list) and tuple(shape) != arry.shape:
                    return False
                # (1,2)
                elif isinstance(shape, tuple|list) and tuple(shape) == arry.shape:
                    pass
                # tuple[int,int]
                elif len(get_args(shape)) != len(arry.shape):
                    return False

            # shape, dtype
            # dtype: bool, int, float
            #        np.bool, ...
            #        np.dtype[type]
            if n_args == 2:
                dtype = self.args[1]
                if len(get_args(dtype)) > 0:
                    dtype = get_args(dtype)[0]

                if dtype == Any:
                    return True
                elif dtype in PY_TYPES and dtype != arry.dtype.type:
                    return False
                elif dtype in NP_TYPES and not issubclass(arry.dtype.type, dtype):
                    return False
            # end
            return True

    IS_INSTANCE_OF["numpy.ndarray"] = IsNumpyArray
except Exception as e:
    pass


# ---------------------------------------------------------------------------
# Optional module: pandas
# ---------------------------------------------------------------------------

try:
    import pandas as pd

    # Check if the type is a DataFrame or a Series
    #
    # For a DataFrame it is possible to check:
    #   1) if there are a list of columns
    #   2) if the columns have a specific type
    #   3) if contains 'at most' a list of columns.
    #      The other columns can be represented by '*'
    #      Simplified: it is checked if contains 'at minimum'
    #      the specified list of columns
    #
    # For a Series it is possible to check:
    #   1) it is of the specified type
    #
    # Syntax:
    #   pd.Series[type]
    #   pd.DataFrame["name1", ...]
    #   pd.DataFrame[["name1", ...]]
    #   pdDataFrame[{"name1": type, ...}]
    #
    #

    # ---------------------------------------------------------------------------
    # __class_getitem__
    # ---------------------------------------------------------------------------

    if not hasattr(pd.Series, "__class_getitem__"):
        @classmethod
        def series_class_getitem(cls, item):
            return typing._GenericAlias(pd.Series, item)


        pd.Series.__class_getitem__ = series_class_getitem

    if not hasattr(pd.DataFrame, "__class_getitem__"):
        @classmethod
        def dataframe_class_getitem(cls, item):
            if not isinstance(item, tuple):
                item = (item,)
            return typing._GenericAlias(pd.DataFrame, item)


        pd.DataFrame.__class_getitem__ = dataframe_class_getitem


    # ---------------------------------------------------------------------------
    # IsPandas
    #   IsSeries
    #   IsDataFrame
    #

    class IsPandas(IsInstance):
        def __init__(self, tp):
            super().__init__(tp)


    class IsSeries(IsPandas):
        def __init__(self, tp):
            super().__init__(tp)

        def is_instance(self, ser: pd.Series):
            if not isinstance(ser, pd.Series):
                return False

            if len(self.args) == 0:
                return True

            base_type = self.args[0]
            ser_dtype = ser.dtype.type
            issc = issubclass(ser_dtype, base_type)

            if issc or len(ser) == 0:
                return True

            # Note: dtype[object_] is a generic type to contain any other object type
            #       The trick is to test just some objects
            if not issubclass(ser_dtype, np.object_):
                return False

            # n = len(ser)
            # for _ in range(10):
            #     i = randrange(n)
            #     val = ser.iloc[i]
            #     if not is_instance(val, base_type):
            #         return False
            if len(ser) > 0:
                val = ser.iloc[0]
                if not is_instance(val, base_type):
                    return False
            return True


    class IsDataFrame(IsPandas):
        def __init__(self, tp):
            super().__init__(tp)
            # self._dtypes = tuple(set(self.args))
            if len(self.args) == 0:
                self._columns = []
                self._coltypes = None
            elif len(self.args) == 1 and isinstance(self.args[0], str):
                self._columns = [self.args[0]]
                self._coltypes = None
            elif len(self.args) == 1 and isinstance(self.args[0], list):
                self._columns = self.args[0]
                self._coltypes = None
            elif len(self.args) == 1 and isinstance(self.args[0], dict):
                self._columns = list(self.args[0].keys())
                self._coltypes = self.args[0]
            elif len(self.args) > 1:
                self._columns = self.args
                self._coltypes = None
            else:
                raise ValueError("Unsupported 'pd.DataFrame[...]' syntax")
            pass

        def is_instance(self, df: pd.DataFrame):
            if not isinstance(df, pd.DataFrame):
                return False
            if len(self._columns) == 0:
                return True

            columns = df.columns
            for col in self._columns:
                if col not in columns:
                    return False

            if self._coltypes is None:
                return True

            for col in columns:
                ctype = self._coltypes[col]
                dtype = df[col].dtype

                if ctype is None:
                    continue
                if issubclass(dtype, ctype):
                    continue
                else:
                    return False

            # df_types = [t.type for t in df.dtypes]
            # df_objects: list[tuple[int, typing.Any]] = []
            #
            # if len(self.args) != len(df_types):
            #     return False
            #
            # for i, dft in enumerate(df_types):
            #     # series of type 'O' require special treatment
            #     if issubclass(dft, np.object_):
            #         df_objects.append((i, dft))
            #         continue
            #     if not issubclass(dft, self._dtypes):
            #         return False
            #
            # # special processing
            # if len(df_objects) == 0:
            #     if not self._check_object_types(df, df_objects):
            #         return False
            return True

        def _check_object_types(self, df: pd.DataFrame, df_objects: list[tuple[int, typing.Any]]):
            # TODO: missing implementation
            for i, dft in enumerate(df_objects):
                pass
            return True


    # end

    IS_INSTANCE_OF['pandas.core.series.Series'] = IsSeries
    IS_INSTANCE_OF['pandas.core.frame.DataFrame'] = IsDataFrame
except Exception as e:
    pass
# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
