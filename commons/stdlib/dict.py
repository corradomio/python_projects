#
# Extends 'builtin.dict' to support:
#
#       d.item      converted into d["item"]
#       d.get("item", default_value)
#       d.get("item1.item2....", default_value)
#           navigate the dictionary
#
# if the value is a 'builtin.dict', it is converted into a 'dict'
#
# class dict(object):
#     @staticmethod # known case
#     def fromkeys(*args, **kwargs)
#     @staticmethod # known case of __new__
#     def __new__(*args, **kwargs)
#
#     def clear(self)
#     def copy(self)
#     def get(self, *args, **kwargs)
#     def items(self)
#     def keys(self)
#     def pop(self, k, d=None)
#     def popitem(self, *args, **kwargs)
#     def setdefault(self, *args, **kwargs)
#     def update(self, E=None, **F)
#     def values(self)
#
#     def __class_getitem__(self, *args, **kwargs)
#     def __contains__(self, *args, **kwargs)
#     def __delitem__(self, *args, **kwargs)
#     def __eq__(self, *args, **kwargs)
#     def __getattribute__(self, *args, **kwargs)
#     def __getitem__(self, y)
#     def __ge__(self, *args, **kwargs)
#     def __gt__(self, *args, **kwargs)
#     def __init__(self, seq=None, **kwargs)
#     def __ior__(self, *args, **kwargs)
#     def __iter__(self, *args, **kwargs)
#     def __len__(self, *args, **kwargs)
#     def __le__(self, *args, **kwargs)
#     def __lt__(self, *args, **kwargs)
#     def __ne__(self, *args, **kwargs)
#     def __or__(self, *args, **kwargs)
#     def __repr__(self, *args, **kwargs)
#     def __reversed__(self, *args, **kwargs)
#     def __ror__(self, *args, **kwargs)
#     def __setitem__(self, *args, **kwargs)
#     def __sizeof__(self)
#     __hash__ = None
from typing import Union

BuiltinDict = dict


class dict(BuiltinDict):

    separator = '.'

    def __init__(self, seq=None, **kwargs):
        if seq is None:
            super().__init__(kwargs)
        else:
            super().__init__(seq, **kwargs)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        # convert builtin_dict into this dict
        if type(value) == BuiltinDict:
            value = dict(value)
            self.__setitem__(item, value)
        return value

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def __class_getitem__(cls, item):
        res = super().__class_getitem__(item)
        return res

    def get(self, key, defval=None):
        """
        Return the value of the specified key or the default value if not present
        It is possible to navigate inside the dictionary using a composite key
        with thw form

            'key1.key2....'

        This means that the keys must not have the '.' in their names

        :param key: the key to select or a list of keys
        :param defval: default value if the key is not present
        :return: the value or the default value
        """
        # if self.__contains__(key):
        #     return self.__getitem__(key)

        parts = key.split(self.separator)
        d = self
        val = defval
        for part in parts:
            if not d.__contains__(part):
                return defval
            val = d.__getitem__(part)
            d = val
        return val

    def set(self, key, value):
        """
        Assign to the key the specified value.
        It is possible to navigate inside the dictionary using a composite key
        with thw form

            'key1.key2....'

        This means that the keys must not have the '.' in their names

        :param key:
        :param value:
        :return: None
        """
        parts = key.split(self.separator)
        key = parts[-1]
        parts = parts[:-1]
        d = self
        for part in parts:
            if not d.__contains__(part):
                d[part] = dict()
            d = d[part]
        d[key] = value
        return None

    def select(self, keys=None, mode='select', defval=None):
        """
        Select a subset of keys
        :param keys: list of keys to select (in alternative to 'exclude')
        :param mode: if to 'select' or to 'exclude'
        :param defval: default value to use
        :return:
        """
        selected = dict()
        if mode == 'select':
            for key in keys:
                selected.set(key, self.get(key, defval))
        elif mode == 'exclude':
            for key in self.keys():
                if key not in keys:
                    selected.set(key, self.get(key, defval))
        else:
            raise ValueError(f"Unsupported mode {mode}. Supported: 'select', 'exclude'")
        return selected

    def contains_keys(self, keys, mode='some'):
        """
        Check if the dictionary contains some key in the list

        :param keys: keys to check
        :param mode: how to check the keys. Available modes: 'some', 'all'
        :return: true if the condition is satisfied, false otherwise
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        if mode == 'some':
            for key in keys:
                if self.__contains__(key):
                    return True
            return False
        elif mode == 'all':
            for key in keys:
                if not self.__contains__(key):
                    return False
            return True
        else:
            raise ValueError(f"Unsupported mode {mode}. Supported: 'some', 'all' ")

    def delete_keys(self, keys):
        """
        Delete one or more keys

        :param keys: list of keys to delete
        :return:None
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for key in keys:
            self.delete(key)

    def delete(self, key):
        """
        Delete a key, if it is present, otherwise it does nothing

        :param key:
        :return: None
        """
        parts = key.split(self.separator)
        key = parts[-1]
        parts = parts[:-1]
        d = self
        for part in parts:
            if not d.__contains__(part):
                return

        if d.__contains__(key):
            d.__delitem__(key)

    def to_list(self):
        """
        Convert the dictionary in a list of pairs.
        Note that:

            list(dict)      -> [key1, ...]
            dict.keys()     -> dict_keys([key1, ...])
            dict.values()   -> dict_values([value1, ...])
            dict.to_list()  -> [(key, value1), ...]

        :return: a list of pairs (key, value)
        """
        return [(key, self.__getitem__(key)) for key in self.keys()]
# end


def reverse_dict(d: Union[BuiltinDict, dict]) -> dict:
    """
    Reverse the dictionary
    :param d: dictionary
    :return: the reversed dictionary
    """
    drev = {d[k]: k for k in d}
    assert len(d) == len(drev)
    return drev


# ---------------------------------------------------------------------------
# Extends 'is_instance'
# ---------------------------------------------------------------------------

from . import is_instance as iii

iii.IS_INSTANCE_OF['stdlib.dict.dict'] = iii.IsDict


# ---------------------------------------------------------------------------
# Extends 'is_instance'
# ---------------------------------------------------------------------------
