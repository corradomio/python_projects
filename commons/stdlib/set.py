# class set(object):
#     def __init__(self, seq=())
#     def add(self, *args, **kwargs)
#     def clear(self, *args, **kwargs)
#     def copy(self, *args, **kwargs)
#     def difference(self, *args, **kwargs)
#     def difference_update(self, *args, **kwargs)
#     def discard(self, *args, **kwargs)
#     def intersection(self, *args, **kwargs)
#     def intersection_update(self, *args, **kwargs)
#     def isdisjoint(self, *args, **kwargs)
#     def issubset(self, *args, **kwargs)
#     def issuperset(self, *args, **kwargs)
#     def pop(self, *args, **kwargs)
#     def remove(self, *args, **kwargs)
#     def symmetric_difference(self, *args, **kwargs)
#     def symmetric_difference_update(self, *args, **kwargs)
#     def union(self, *args, **kwargs)
#     def update(self, *args, **kwargs)
#     def __and__(self, *args, **kwargs)
#     def __class_getitem__(self, *args, **kwargs)
#     def __contains__(self, y)
#     def __eq__(self, *args, **kwargs)
#     def __getattribute__(self, *args, **kwargs)
#     def __ge__(self, *args, **kwargs)
#     def __gt__(self, *args, **kwargs)
#     def __iand__(self, *args, **kwargs)
#     def __ior__(self, *args, **kwargs)
#     def __isub__(self, *args, **kwargs)
#     def __iter__(self, *args, **kwargs)
#     def __ixor__(self, *args, **kwargs)
#     def __len__(self, *args, **kwargs)
#     def __le__(self, *args, **kwargs)
#     def __lt__(self, *args, **kwargs)
#     @staticmethod # known case of __new__
#     def __new__(*args, **kwargs)
#     def __ne__(self, *args, **kwargs)
#     def __or__(self, *args, **kwargs)
#     def __rand__(self, *args, **kwargs)
#     def __reduce__(self, *args, **kwargs)
#     def __repr__(self, *args, **kwargs)
#     def __ror__(self, *args, **kwargs)
#     def __rsub__(self, *args, **kwargs)
#     def __rxor__(self, *args, **kwargs)
#     def __sizeof__(self)
#     def __sub__(self, *args, **kwargs)
#     def __xor__(self, *args, **kwargs)
#     __hash__ = None

BuiltinSet = set


class set(BuiltinSet):
    def __init__(self, seq=()):
        super().__init__(seq)

    def add(self, __element) -> BuiltinSet:
        super().add(__element)
        return self

    def clear(self):
        super().clear()
        return self
