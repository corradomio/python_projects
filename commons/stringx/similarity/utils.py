from math import sqrt


def check_params(s0, s1):
    if s0 is None:
        raise TypeError("Argument s0 is NoneType.")
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")


def zerovec(n):
    return [0 for j in range(n)]


def zeromat(n, m):
    return [[0 for j in range(m)] for i in range(n)]


def sq(x): return x*x
def sqr(x): return sqrt(x)


def cosdist(v0, v1):
    snum = 0
    den0 = 0
    den1 = 0
    for e in v0:
        e0 = v0[e]
        e1 = v1[e]
        snum += e0 * e1
        den0 += sq(e0)
        den1 += sq(e1)
    # end
    return snum / (sqr(den0) * sqr(den1))


class bag(object):

    def __init__(self, seq=()):
        self._bag = dict()

        if isinstance(seq, bag):
            for e in seq:
                self.add(e, count=seq.get(e))
        else:
            for e in seq:
                self.add(e, count=1)

    def add(self, e, count=1):
        if count <= 0:
            pass
        elif e in self._bag:
            self._bag[e] += count
        else:
            self._bag[e] = count

    def clear(self):
        self._bag.clear()

    def copy(self):
        return bag(self)

    def get(self, e):
        return self._bag.get(e, 0)

    def count(self):
        c = 0
        for e in self._bag:
            c += self._bag[e]
        return c

    def difference(self, other):
        diff = bag()
        for e in self._bag:
            diff.add(e, count=self._bag[e])
        for e in other:
            diff.discard(e, count=other.get(e))
        return diff

    def difference_update(self, other):
        for e in other:
            self.discard(e, count=other.get(e))

    def discard(self, e, count=1):
        if e not in self._bag:
            pass
        elif count >= self._bag[e]:
            del self._bag[e]
        else:
            self._bag[e] -= count

    def intersection(self, other):
        i = bag()
        for e in self._bag:
            if e in other:
                i.add(e, min(self._bag[e], other.get(e)))
        return i

    def intersection_update(self, other):
        i = self.intersection(other)
        self.clear()
        self.update(i)

    def isdisjoint(self, other):
        for e in self._bag:
            if e in other:
                return False
        return  True

    def issubbag(self, other):
        for e in self._bag:
            if e not in other:
                return  False
            if self._bag[e] > other.get(e):
                return False
        return True

    def issuperbag(self, other):
        return other.issubbag(self)

    def issamebag(self, other):
        if len(self) != len(other):
            return False
        for e in self._bag:
            if e not in other:
                return False
            if self._bag[e] != other.get(e):
                return False
        return True

    def pop(self):
        for e in self._bag:
            del self._bag[e]
            return e
        raise KeyError("pop from an empty bag")

    def remove(self, e, count=1):
        if e not in self._bag:
            raise KeyError(e)
        else:
            self.discard(e, count=count)

    def symmetric_difference(self, other):
        sd = bag()
        for e in self._bag:
            sd.add(e, self.get(e) - other.get(e))
        for e in other:
            sd.add(e, other.get(e) - self.get(e))
        return sd

    def symmetric_difference_update(self, other):
        sdiff = self.symmetric_difference(other)
        self.clear()
        self.update(sdiff)

    def union(self, other):
        u = bag()
        for e in self._bag:
            u.add(e, count=self.get(e))
        for e in other:
            u.add(e, count=other.get(e))
        return u

    def update(self, other):
        if isinstance(other, bag):
            for e in other:
                self.add(e, count=other.get(e))
        else:
            for e in other:
                self.add(e)

    def keys(self):
        return self._bag.keys()

    def __and__(self, other):
        return self.intersection(other)

    def __rand__(self, other):
        return self.intersection(other)

    def __getitem__(self, e):
        return self._bag[e]

    def __contains__(self, e):
        return e in self._bag

    def __eq__(self, other):
        return self.issamebag(other)

    def __ge__(self, other):
        return self.issuperbag(other)

    def __gt__(self, other):
        return self.issuperbag(other) and not self.issamebag(other)

    def __iand__(self, other):
        self.intersection_update(other)
        return self

    def __ior__(self, other):
        self.update(other)
        return self

    def __isub__(self, other):
        self.difference_update(other)
        return self

    def __iter__(self):
        return self._bag.__iter__()

    def __len__(self, *args, **kwargs):
        return self._bag.__len__()

    def __le__(self, other):
        return self.issubbag(other)

    def __lt__(self, other):
        return self.issubbag(other) and not self.issamebag(other)

    def __ne__(self, other):
        return not self.issamebag(other)

    def __or__(self, other):
        return self.union(other)

    def __ror__(self, other):
        return self.union(other)

    def __repr__(self):
        return self._bag.__repr__()

    def __sub__(self, other):
        return self.difference(other)

    def __rsub__(self, other):
        return other.difference(self)

    def __hash__(self):
        return hash(self._bag)
