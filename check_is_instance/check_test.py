from typing import Collection, Union, Mapping, Dict, Sequence

from stdlib.is_instance import is_instance

# assert is_instance("a", [0, "a"])
# assert (is_instance([1, 2., 'tre'], Collection[Union[int, float, str]]))
# assert (is_instance({'one': 1}, Mapping[str, int]))
assert (is_instance([1., 2., 3.], Sequence))