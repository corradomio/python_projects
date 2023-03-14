
class bag(dict):
    def __init__(self, l=None):
        super().__init__()
        if isinstance(l, (tuple, list)):
            for e in l:
                self.add(e)
    # end

    def at(self, at: int) -> str:
        return list(self.keys())[at]

    def add(self, e, count=1):
        assert count >= 0
        if e in self:
            self[e] += count
        else:
            self[e] = count

    def get(self, e) -> int:
        return self[e] if e in self else 0

    def count(self):
        c = 0
        for e in self:
            c += self[e]
        return c

    def __or__(self, other):
        return self.union(other)

    def union(self, other: 'bag') -> 'bag':
        assert isinstance(other, bag)
        r = bag()
        for e in self:
            r.add(e, self[e])
        for e in other:
            r.add(e, other[e])
        return r
    # end

    def __sub__(self, other):
        return self.difference(other)

    def difference(self, other: 'bag') -> 'bag':
        assert isinstance(other, bag)
        r = bag()
        for e in other:
            if e not in self:
                continue
            if self[e] <= other[e]:
                continue
            else:
                r.add(e, self[e ] -other[e])
        return r
    # end

    def __and__(self, other):
        return self.intersection(other)

    def intersection(self, other: 'bag') -> 'bag':
        assert isinstance(that, bag)
        r = bag()
        for e in other:
            if e not in self:
                continue
            else:
                r.add(e, min(self[e], other[e]))
        return r
    # end
# end
