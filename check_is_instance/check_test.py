from numbers import Number
from typing import Collection, Union, Mapping, Dict, Sequence, get_origin, get_args, Literal, Type

from stdlib.is_instance import is_instance, All, Immutable, Supertype

# assert is_instance("a", [0, "a"])
# assert (is_instance([1, 2., 'tre'], Collection[Union[int, float, str]]))
# assert (is_instance({'one': 1}, Mapping[str, int]))
# assert (is_instance([1., 2., 3.], Sequence))

# 'isinstance' DOES NOT SUPPORT this syntax!!
# assert isinstance([1,2,3], list[int])

# assert is_instance("a", Literal["a", "b", "c"])

# assert is_instance(1,Number)
# assert is_instance(1,int)
# assert is_instance(1,All[Number,int])

# assert is_instance((1,2), Immutable)



class Base:
    pass

class Derived(Base):
    pass

class Child(Derived):
    pass


# c = Child()
# assert is_instance(c, Child)
# assert is_instance(c, Type[Child])
#
# assert is_instance(c, Derived)
# assert is_instance(c, Type[Derived])
#
# assert is_instance(c, Base)
# assert is_instance(c, Type[Base])

b = Base()
assert is_instance(b, Supertype[Child])

assert isinstance([1], Sequence)