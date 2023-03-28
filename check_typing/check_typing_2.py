from typing import Union, Optional, Any, Set

from is_instance import is_instance

# d = {'a': 1, 'b': 2, 'c': None}

# print(is_instance(d, dict[str, Optional[Any]]))

# print(is_instance([1, 2, None], list[Optional[int]]))

print(is_instance({1, 2, 3}, Set[int]))
