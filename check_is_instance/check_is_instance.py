from typing import *
from types import *
from collections import *
from stdlib.is_instance import is_instance, All, Immutable, Const, Literals


def ispositive(value):
    return is_instance(value, int) and value > 0

VALUES = ['a', 'b', 'c']
assert (is_instance(1, Union[int, Literals[VALUES]]))
assert (is_instance('a', Union[int, Literals[VALUES]]))
assert (not is_instance('d', Union[int, Literals[VALUES]]))

assert is_instance(1, ispositive)

print(is_instance(1, int | float))
