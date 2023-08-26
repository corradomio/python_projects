from builtins import dict


# class dict(object):
#     def clear(self):  # real signature unknown; restored from __doc__
#     def copy(self):  # real signature unknown; restored from __doc__
#     def get(self, *args, **kwargs):  # real signature unknown
#     def items(self):  # real signature unknown; restored from __doc__
#     def keys(self):  # real signature unknown; restored from __doc__
#     def pop(self, k, d=None):  # real signature unknown; restored from __doc__
#     def popitem(self, *args, **kwargs):  # real signature unknown
#     def setdefault(self, *args, **kwargs):  # real signature unknown
#     def update(self, E=None, **F):  # known special case of dict.update
#     def values(self):  # real signature unknown; restored from __doc__
#
#     def __class_getitem__(self, *args, **kwargs):  # real signature unknown
#     def __contains__(self, *args, **kwargs):  # real signature unknown
#     def __delitem__(self, *args, **kwargs):  # real signature unknown
#     def __eq__(self, *args, **kwargs):  # real signature unknown
#     def __getattribute__(self, *args, **kwargs):  # real signature unknown
#     def __getitem__(self, y):  # real signature unknown; restored from __doc__
#     def __ge__(self, *args, **kwargs):  # real signature unknown
#     def __gt__(self, *args, **kwargs):  # real signature unknown
#     def __init__(self, seq=None, **kwargs):  # known special case of dict.__init__
#     def __ior__(self, *args, **kwargs):  # real signature unknown
#     def __iter__(self, *args, **kwargs):  # real signature unknown
#     def __len__(self, *args, **kwargs):  # real signature unknown
#     def __le__(self, *args, **kwargs):  # real signature unknown
#     def __lt__(self, *args, **kwargs):  # real signature unknown
#     def __ne__(self, *args, **kwargs):  # real signature unknown
#     def __or__(self, *args, **kwargs):  # real signature unknown
#     def __repr__(self, *args, **kwargs):  # real signature unknown
#     def __reversed__(self, *args, **kwargs):  # real signature unknown
#     def __ror__(self, *args, **kwargs):  # real signature unknown
#     def __setitem__(self, *args, **kwargs):  # real signature unknown
#     def __sizeof__(self):  # real signature unknown; restored from __doc__
#     __hash__ = None


class bag(dict):

    def __init__(self, seq=()):
        super().__init__()

        if isinstance(seq, bag):
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
        self.set(e, count + self.get(e))

    def discard(self, e, count=1):
        self.add(e, -count)

    def remove(self, e):
        if e not in self:
            raise KeyError(e)
        else:
            del self[e]

    def count(self) -> int:
        c = 0
        for e in self:
            c += self[e]
        return c
    
    def union_update(self, that):
        assert isinstance(that, bag)
        
        for e in that:
            self.set(e, self.get(e) + that[e])
        return self

    def update(self, that):
        self.union_update(that)
    
    def intersection_update(self, that):
        assert isinstance(that, bag)

        for e in that:
            self.set(e, min(self.get(e), that[e]))
        return self
    
    def difference_update(self, that):
        assert isinstance(that, bag)
        
        for e in that:
            self.set(e, self.get(e) - that[e])
        return self
            
    def symmetric_difference_update(self, that):
        assert isinstance(that, bag)
        this = bag(self)

        self.clear()
        self.union_update(this.difference(that))
        self.union_update(that.difference(this))
        return self

    def union(self, that) -> "bag":
        u = bag(self)
        u.union_update(that)
        return u
    
    def intersection(self, that) -> "bag":
        i = bag(self)
        i.intersection_update(that)
        return i
    
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

    def __and__(self, that) -> "bag":
        return self.intersection(that)

    def __rand__(self, that) -> "bag":
        return that.intersection(self)

    def __iand__(self, that) -> "bag":
        return self.intersection_update(that)

    def __or__(self, that) -> "bag":
        return self.union(that)

    def __ror__(self, that) -> "bag":
        return that.union(self)

    def __ior__(self, that) -> "bag":
        return self.union_update(that)

    def __sub__(self, that) -> "bag":
        return self.difference(that)

    def __rsub__(self, that) -> "bag":
        return that.difference(self)

    def __isub__(self, that) -> "bag":
        return self.difference_update(that)

