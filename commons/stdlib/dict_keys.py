#
# A dictionary accepting ONLY a predefined list of keywords
#
from typing import Union

from .is_instance import is_instance


class dict_keys(dict):
    def __init__(self, seq=None, *, keys: Union[list[str], set[str]]):
        assert is_instance(keys, Union[list[str], set[str]])
        self._keys = set(keys)
        if seq is None:
            super().__init__()
        else:
            super().__init__(seq)

    def __getitem__(self, item):
        if item not in self._keys: raise ValueError(f"Item {item} not a valid key")
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        if item not in self._keys: raise ValueError(f"Item {item} not a valid key")
        return super().__setitem__(item, value)

    def __getattr__(self, item):
        if item == '_keys':
            return super().__getattribute__(item)
        return self.__getitem__(item)

    def __setattr__(self, item, value):
        if item == '_keys':
            return super().__setattr__(item, value)
        return self.__setitem__(item, value)

    def __delattr__(self, item):
        return self.__delitem__(item)

# end
