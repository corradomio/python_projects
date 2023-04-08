from typing import *
from collections import *
from is_instance import is_instance, All


print(is_instance({1, 2, 3}, set))
print(is_instance({1, 2, 3}, Set))
print(is_instance({1, 2, 3}, set[int]))
print(is_instance({1, 2, 3}, Set[int]))

print(is_instance([1, 2, 3], list))
print(is_instance([1, 2, 3], List))
print(is_instance([1, 2, 3], list[int]))
print(is_instance([1, 2, 3], List[int]))

print(is_instance((1, 2, 3), tuple))
print(is_instance((1, 2, 3), Tuple))
print(is_instance((1, 2, 3), tuple[int]))
print(is_instance((1, 2, 3), Tuple[int]))
print("false", is_instance((1, 2, 3), Tuple[int, int]))
print(is_instance((1, 2, 3), Tuple[int, int, int]))

print(is_instance(deque([1, 2, 3]), deque))
# print(is_instance(deque([1, 2, 3]), deque[int]))
print(is_instance(deque([1, 2, 3]), Deque[int]))

dd = defaultdict()
dd[1] = '1'
print(is_instance(dd, defaultdict))
# print(is_instance(dd, defaultdict[int, str]))
print(is_instance(dd, DefaultDict[int, str]))

print(is_instance(lambda x: x, Callable))
print(is_instance(1, (type(None), str, float, bytes, int)))


Vector = list[float]
Vector = NewType('Vector', list[float])
print(is_instance([1., 2., 3.], Vector))
print(is_instance([1., 2., 3.], Sequence))

print(is_instance({1., 2., 3.}, Container))
print(is_instance({1., 2., 3.}, Iterable))
print("false", is_instance([1., 2., 3.], Hashable))
print(is_instance({1., 2., 3.}, Sized))
print(is_instance(iter({1., 2., 3.}), Iterator))
print("false", is_instance({1., 2., 3.}, Iterator))
print(is_instance([1., 2., 3.], Reversible))

print(is_instance('ciccio', Union[None, type(None), int, float, bytes, str]))

class C:
    def __contains__(self, item):
        pass
    def __iter__(self):
        pass
    def __next__(self):
        pass
    def __hash__(self):
        pass
    def __len__(self):
        pass
    def __reversed__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass
    pass

class D:
    def __contains__(self, item):
        pass
    def __iter__(self):
        pass

c = C()
d = D()

print(is_instance(c, Container))
print(is_instance(c, Iterable))
print(is_instance(c, Hashable))
print(is_instance(c, Sized))
print(is_instance(c, Iterator))
print(is_instance(c, Reversible))
print(is_instance(c, Callable))
print("false", is_instance(d, All[Iterable, Sized]))
print(is_instance(d, All[Iterable, Container]))
print(is_instance(frozenset([1,2]), frozenset))

print(is_instance([1, 2., 'tre'], Collection))
print(is_instance([1, 2., 'tre'], Collection[Union[int, float, str]]))
print(is_instance((1, 2., 'tre'), Collection))
print(is_instance((1, 2., 'tre'), Collection[Union[int, float, str]]))
