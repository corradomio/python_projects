from typing import Collection, Sequence, assert_type, reveal_type

from stdlib import is_instance

# assert (is_instance({1, 2, 3}, Collection[int]))
assert (not is_instance([1, 2, 3], Collection[int]))
print(reveal_type([1,2,3]))
