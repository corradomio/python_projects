from typing import *
from collections import *
from is_instance import is_instance, All, Const


is_instance(1, Const[int])

is_instance([1, 2, 3], list[int])
is_instance(deque([1, 2, 3]), deque[int])

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
assert(is_instance(p, Point))


class C:
    def __int__(self):
        pass

    def method(self, p1: int) -> int:
        let: Const[int] = 0
