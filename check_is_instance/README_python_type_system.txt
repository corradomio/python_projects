Python operators
----------------

    isinstance
    issubclass


Python modules
--------------

    typing
    types
    collections.abc
    collections


'typing' imports:

    import abc                  it is NOT 'collections.abc'
    import collections
    import collections.abc
    import copyreg
    import functools
    import operator
    import sys
    import types


Note: 'abc' is 'Abstract Base Class' and it is available in
several places.


type alias:

    from typing import TypeAlias
    Vector: TypeAlias = list[float]

    'Vector' is equivalent/the same than 'list[float]'

new type

    rom typing import NewType
    UserId = NewType('UserId', int)

    'UserId' is a SUBTYPE than 'int'

useful functions

    typing.get_origin(tp)   X [Y,Z,...] -> X
    typing.get_args(tp)     X [Y,Z,...] -> [Y,Z,...]