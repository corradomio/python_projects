from typing import Collection

from stdlib import is_instance

# assert (is_instance({1, 2, 3}, Collection[int]))
assert (not is_instance([1, 2, 3], Collection[int, int]))