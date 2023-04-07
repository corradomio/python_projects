from typing import Set, List, Tuple

from is_instance import is_instance


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
print(is_instance((1, 2, 3), Tuple[int, int]))
print(is_instance((1, 2, 3), Tuple[int, int, int]))
