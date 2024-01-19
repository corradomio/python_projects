from builtins import dict
from collections import Counter

# Note:
#   Counter(dict) has similar behavior
#       __init__
#       __missing__
#       most_common
#       elements
#       update
#       subtract
#       copy
#       __add__     __iadd__
#       __sub__     __isub__
#       __or__      __ior__
#       __and__     __iand__
#       __pos__
#       __neg__
#


# class set(object):
#     def copy(self, *args, **kwargs):
#
#     def remove(self, *args, **kwargs):
#     def clear(self, *args, **kwargs):
#
#     def add(self, *args, **kwargs):
#     def discard(self, *args, **kwargs):
#
#     def union(self, *args, **kwargs):
#     def update(self, *args, **kwargs):
#     def intersection(self, *args, **kwargs):
#     def intersection_update(self, *args, **kwargs):
#     def difference(self, *args, **kwargs):
#     def difference_update(self, *args, **kwargs):
#     def symmetric_difference(self, *args, **kwargs):
#     def symmetric_difference_update(self, *args, **kwargs):
#
#     def isdisjoint(self, *args, **kwargs):
#     def issubset(self, *args, **kwargs):
#     def issuperset(self, *args, **kwargs):
#
#     def pop(self, *args, **kwargs):
#
#     def __and__(self, *args, **kwargs):
#     def __rand__(self, *args, **kwargs):
#     def __iand__(self, *args, **kwargs):
#     def __or__(self, *args, **kwargs):
#     def __ror__(self, *args, **kwargs):
#     def __ior__(self, *args, **kwargs):
#     def __sub__(self, *args, **kwargs):
#     def __rsub__(self, *args, **kwargs):
#     def __isub__(self, *args, **kwargs):
#     def __xor__(self, *args, **kwargs):
#     def __rxor__(self, *args, **kwargs):
#     def __ixor__(self, *args, **kwargs):
#
#     def __contains__(self, y):
#     def __eq__(self, *args, **kwargs):
#     def __ge__(self, *args, **kwargs):
#     def __gt__(self, *args, **kwargs):
#     def __le__(self, *args, **kwargs):
#     def __lt__(self, *args, **kwargs):
#     def __ne__(self, *args, **kwargs):
#
#     def __init__(self, seq=()):
#     def __iter__(self, *args, **kwargs):
#     def __len__(self, *args, **kwargs):
#
#     def __getattribute__(self, *args, **kwargs):
#     def __class_getitem__(self, *args, **kwargs):
#     def __reduce__(self, *args, **kwargs):
#     def __sizeof__(self):
#     def __repr__(self, *args, **kwargs):


# class dict(object):
#     def clear(self):
#     def copy(self):
#
#     def items(self):
#     def keys(self):
#     def values(self):
#
#     def get(self, *args, **kwargs):
#     def pop(self, k, d=None):
#     def popitem(self, *args, **kwargs):
#     def setdefault(self, *args, **kwargs):
#
#     def update(self, E=None, **F):
#
#     def __contains__(self, *args, **kwargs):
#     def __eq__(self, *args, **kwargs):
#     def __ge__(self, *args, **kwargs):
#     def __gt__(self, *args, **kwargs):
#     def __le__(self, *args, **kwargs):
#     def __lt__(self, *args, **kwargs):
#     def __ne__(self, *args, **kwargs):
#
#     def __or__(self, *args, **kwargs):
#     def __ror__(self, *args, **kwargs):
#     def __ior__(self, *args, **kwargs):
#
#     def __delitem__(self, *args, **kwargs):
#     def __getitem__(self, y):
#     def __setitem__(self, *args, **kwargs):
#
#     def __init__(self, seq=None, **kwargs):
#     def __iter__(self, *args, **kwargs):
#     def __len__(self, *args, **kwargs):
#     def __repr__(self, *args, **kwargs):
#     def __reversed__(self, *args, **kwargs):
#
#     def __getattribute__(self, *args, **kwargs):
#     def __class_getitem__(self, *args, **kwargs):
#     def __sizeof__(self):

# Note: in theory:
#
#       a & b  ::=  for e in b: r[e] = min(a[e], b[e])
#
#       a | b  ::=  for e in a: r[e] = max(a[e], b[e])
#                   for e in b: r[e] = max(a[e], b[e])
#
#       a + b  :=   for e in a: r[e] = a[e] + b[e]
#                   for e in b: r[e] = a[e] + b[e]
#
#       a - b  :=   for e in a: r[e] = a[e] - b[e]
#                   for e in b: r[e] = a[e] - b[e]

class bag(dict):

    def __init__(self, seq=()):
        super().__init__()

        if isinstance(seq, dict):
            for e in seq:
                self.add(e, count=seq[e])
        else:
            for e in seq:
                self.add(e, count=1)

    def copy(self) -> "bag":
        return bag(self)

    def get(self, e) -> int:
        return super().get(e, 0)

    def set(self, e, count=1):
        self[e] = count
        if self[e] <= 0:
            del self[e]

    def add(self, e, count=1):
        self.set(e, self.get(e) + count)

    def discard(self, e, count=1):
        self.set(e, self.get(e) - count)

    def remove(self, e):
        del self[e]

    def count(self) -> int:
        """
        Count the number of elements in the bag, considering their multiplicities
        :return: n of elements in the bag
        """
        c = 0
        for e in self:
            c += self[e]
        return c

    #
    # operations
    #

    def update(self, that):
        self.union_update(that)

    def union_update(self, that) -> "bag":
        assert isinstance(that, bag)

        for e in that:
            self.set(e, max(self.get(e), that[e]))
        return self

    def intersection_update(self, that) -> "bag":
        assert isinstance(that, bag)

        elts = list(self.keys())
        for e in elts:
            if e not in that:
                del self[e]

        for e in that:
            self.set(e, min(self.get(e), that[e]))

        return self

    def sum_update(self, that) -> "bag":
        assert isinstance(that, bag)

        for e in that:
            self.set(e, self.get(e) + that[e])
        return self

    def difference_update(self, that) -> "bag":
        assert isinstance(that, bag)

        for e in that:
            self.set(e, self.get(e) - that[e])
        return self

    def symmetric_difference_update(self, that) -> "bag":
        assert isinstance(that, bag)
        this = bag(self)

        self.clear()
        self.union_update(this.difference(that))
        self.union_update(that.difference(this))
        return self

    #
    # operations
    #

    def union(self, that) -> "bag":
        u = bag(self)
        u.union_update(that)
        return u

    def intersection(self, that) -> "bag":
        i = bag(self)
        i.intersection_update(that)
        return i

    def sum(self, that) -> "bag":
        d = bag(self)
        d.sum_update(that)
        return d

    def difference(self, that) -> "bag":
        d = bag(self)
        d.difference_update(that)
        return d

    def symmetric_difference(self, that) -> "bag":
        this = self
        s = bag()
        s.union_update(this.difference(that))
        s.union_update(that.difference(this))
        return s

    #
    # predicates
    #

    def isdisjoint(self, that) -> bool:
        for e in self:
            if e in that:
                return False
        return True

    def issubbag(self, that) -> bool:
        for e in self:
            if e not in that:
                return False
            if self[e] > that.get(e):
                return False
        return True

    def issuperbag(self, that) -> bool:
        return that.issubbag(self)

    def issamebag(self, that) -> bool:
        if len(self) != len(that):
            return False
        for e in self:
            if self[e] != that.get(e):
                return False
        return True

    #
    #
    #

    def __missing__(self, e):
        return 0

    #
    # predicates
    #

    def __eq__(self, that):
        return self.issamebag(that)

    def __ne__(self, that):
        return not self.issamebag(that)

    def __le__(self, that):
        return self.issubbag(that)

    def __lt__(self, that):
        return self.issubbag(that) and not self.issamebag(that)

    def __ge__(self, that):
        return self.issuperbag(that)

    def __gt__(self, that):
        return self.issuperbag(that) and not self.issamebag(that)

    #
    # operations:
    #   a & b, a | b
    #   a ^ b (symmetric difference)
    #   a + b, a - b
    #

    # a & b, a &= b
    def __and__(self, that) -> "bag":
        return self.intersection(that)

    def __iand__(self, that) -> "bag":
        return self.intersection_update(that)

    # def __rand__(self, that) -> "bag":
    #     return that.intersection(self)

    # a | b, a |= b
    def __or__(self, that) -> "bag":
        return self.union(that)

    def __ior__(self, that) -> "bag":
        return self.union_update(that)

    # def __ror__(self, that) -> "bag":
    #     return that.union(self)

    # a ^ b, a ^= b
    def __xor__(self, that):
        return self.symmetric_difference(that)

    def __ixor__(self, that):
        return self.symmetric_difference_update(that)

    # def __rxor__(self, that):
    #     return that.symmetric_difference(self)

    # a + b, a += b
    def __add__(self, that) -> "bag":
        return self.sum(that)

    def __iadd__(self, that) -> "bag":
        return self.sum_update(that)

    # def __radd__(self, that) -> "bag":
    #     return that.union(self)

    # a - b, a -= b
    def __sub__(self, that) -> "bag":
        return self.difference(that)

    def __isub__(self, that) -> "bag":
        return self.difference_update(that)

    # def __rsub__(self, that) -> "bag":
    #     return that.difference(self)

# end
