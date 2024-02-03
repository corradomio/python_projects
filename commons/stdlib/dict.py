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

BuiltinDict = dict


class dict(dict):

    def __init__(self, seq=None, **kwargs):
        if seq is None:
            super().__init__(kwargs)
        else:
            super().__init__(seq, **kwargs)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, BuiltinDict):
            value = dict(value)
        return value

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def get(self, key, defval=None):
        parts = key.split('.')
        d = self
        val = defval
        for part in parts:
            if not d.__contains__(part):
                return defval
            val = d.__getitem__(part)
            d = val
        return val
# end
