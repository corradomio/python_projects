from typing import *
from types import *
from collections import *
from stdlib.is_instance import is_instance, All, Immutable, Const, Literals, has_methods, Mapping


class C:
    def method(self):
        pass

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

    def __iter__(self) -> "Self":
        pass


def test_literal():
    assert (not is_instance(1, Literal[2]))
    assert (is_instance(1, Literal[1, 2]))
    assert (is_instance("on", Literal["on", "off"]))


def test_immutable():
    assert (is_instance(1, Immutable))
    assert (is_instance((1, 'str'), Immutable))
    assert (is_instance(frozenset([1, 2, 3]), Immutable))


# def test_final():
#     assert (is_instance(1, Final))
#     assert (is_instance(1, Final[int]))


def test_const():
    assert (is_instance(1, Const))
    assert (is_instance(1, Const[int]))


def test_function():
    assert (is_instance(test_function, FunctionType))
    assert (is_instance(test_function, LambdaType))


def test_method():
    c = C()
    assert (is_instance(c.method, MethodType))


def test_set():
    assert (is_instance({1, 2, 3}, set))
    assert (is_instance({1, 2, 3}, Set))
    assert (is_instance({1, 2, 3}, set[int]))
    assert (is_instance({1, 2, 3}, Set[int]))


def test_list():
    assert (is_instance([1, 2, 3], list))
    assert (is_instance([1, 2, 3], List))
    assert (is_instance([1, 2, 3], list[int]))
    assert (is_instance([1, 2, 3], List[int]))


def test_tuple():
    assert (is_instance((1, 2, 3), tuple))
    assert (is_instance((1, 2, 3), Tuple))
    assert (is_instance((1, 2, 3), tuple[int]))
    assert (is_instance((1, 2, 3), Tuple[int]))
    assert (not is_instance((1, 2, 3), Tuple[int, int]))
    assert (is_instance((1, 2, 3), Tuple[int, int, int]))


def test_collection():
    assert (is_instance((1, 2, 3), Collection))
    assert (is_instance([1, 2, 3], Collection))
    assert (is_instance({1, 2, 3}, Collection))
    assert (is_instance(frozenset({1, 2, 3}), Collection))
    assert (is_instance(deque({1, 2, 3}), Collection))

    assert (is_instance((1, 2, 3), Collection[int]))
    assert (is_instance([1, 2, 3], Collection[int]))
    assert (is_instance({1, 2, 3}, Collection[int]))
    assert (is_instance(frozenset((1, 2, 3)), Collection[int]))
    assert (is_instance(deque((1, 2, 3)), Collection[int]))

    assert (not is_instance([1, 2, 3], Collection[int, int]))
    assert (is_instance((1, 2, 3), Collection[int, int, int]))


def test_deque():
    assert (is_instance(deque([1, 2, 3]), deque))
    assert (is_instance(deque([1, 2, 3]), deque[int]))
    assert (is_instance(deque([1, 2, 3]), Deque[int]))


def test_defaultdict():
    dd = defaultdict()
    dd[1] = '1'
    assert (is_instance(dd, defaultdict))
    assert (is_instance(dd, defaultdict[int, str]))
    assert (is_instance(dd, DefaultDict[int, str]))


def test_namedtuple():
    # namedtuple is a function NOT a type
    Point = namedtuple('Point', ['x', 'y'])
    p = Point(1, 2)
    assert (is_instance(p, Point))


def test_callable():
    assert (is_instance(lambda x: x, Callable))
    assert (is_instance(1, (type(None), str, float, bytes, int)))


def test_vector_alais():
    Vector = list[float]
    assert (is_instance([1., 2., 3.], Vector))
    assert (is_instance([1., 2., 3.], Sequence))


def test_vector_seq():
    Vector = list[float]
    Vector = NewType('Vector', list[float])
    assert (is_instance([1., 2., 3.], Vector))
    assert (is_instance([1., 2., 3.], Sequence))


def test_container():
    assert (is_instance({1., 2., 3.}, Container))
    assert (is_instance({1., 2., 3.}, Iterable))
    assert (not is_instance([1., 2., 3.], Hashable))
    assert (is_instance({1., 2., 3.}, Sized))
    assert (is_instance(iter({1., 2., 3.}), Iterator))
    assert (not is_instance({1., 2., 3.}, Iterator))
    assert (is_instance([1., 2., 3.], Reversible))


def test_collection():
    assert (is_instance([1, 2., 'tre'], Collection))
    assert (is_instance([1, 2., 'tre'], Collection[Union[int, float, str]]))
    assert (is_instance([1, 2., 'tre'], List[Union[int, float, str]]))
    assert (is_instance([1, 2., 'tre'], list[Union[int, float, str]]))
    assert (is_instance((1, 2., 'tre'), Collection))
    assert (is_instance((1, 2., 'tre'), Collection[Union[int, float, str]]))
    assert (is_instance((1, 2., 'tre'), Tuple))
    assert (is_instance((1, 2., 'tre'), Tuple[Union[int, float, str]]))
    assert (is_instance((1, 2., 'tre'), tuple))
    assert (is_instance((1, 2., 'tre'), tuple[Union[int, float, str]]))


def test_mapping():
    # Mapping
    assert (is_instance({'one': 1}, Mapping[str, int]))


def test_union():
    assert (is_instance(None, Union[None, type(None), int, float, bytes, str]))
    assert (is_instance(1, Union[None, type(None), int, float, bytes, str]))
    assert (is_instance(1.1, Union[None, type(None), int, float, bytes, str]))
    assert (is_instance(bytes("ciccio", "utf-8"), Union[None, type(None), int, float, bytes, str]))
    assert (is_instance('ciccio', Union[None, type(None), int, float, bytes, str]))


def test_class():
    c = C()
    d = D()
    assert (is_instance(c, Container))
    assert (is_instance(c, Iterable))
    assert (is_instance(c, Hashable))
    assert (is_instance(c, Sized))
    assert (is_instance(c, Iterator))
    assert (is_instance(c, Reversible))
    assert (is_instance(c, Callable))
    assert (not is_instance(d, All[Iterable, Sized]))
    assert (is_instance(d, All[Iterable, Container]))


def test_frozenset():
    assert (is_instance(frozenset([1, 2]), frozenset))
    assert (is_instance(frozenset([1, 2]), frozenset[int]))


def test_literals():
    VALUES = ('a', 'b', 'c')
    assert (is_instance(1, Union[int, Literals[VALUES]]))
    assert (is_instance('a', Union[int, Literals[VALUES]]))
    assert (is_instance('a', VALUES))


def test_literal_extended():
    assert is_instance("a", (0, "a"))
    assert is_instance(0, (0, "a"))
    assert not is_instance(1, (0, "a"))
    assert is_instance("a", [0, "a"])
    assert is_instance(0, [0, "a"])
    assert not is_instance(1, [0, "a"])


def test_has_methods():
    class C:
        def fit(self,X, y):
            pass
        def predict(self, X):
            pass
        def score(self, X,y):
            pass

    assert has_methods(C, ['fit', 'predict', 'score'])
    assert is_instance(C, has_methods(['fit', 'predict', 'score']))