from typing import Union, Optional, Any, Set
import types

from is_instance import is_instance

# d = {'a': 1, 'b': 2, 'c': None}

# print(is_instance(d, dict[str, Optional[Any]]))

# print(is_instance([1, 2, None], list[Optional[int]]))

# print(is_instance({1, 2, 3}, Set[int]))
print(is_instance(3., Union[int, float, str, None]))
