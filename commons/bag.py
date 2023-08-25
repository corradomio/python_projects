from builtins import set


class bag(object):
    """
    bag() -> new empty bag object
    bag(iterable) -> new bag object

    Build a counted collection of unique elements.
    """

    def __init__(self, seq=()):
        self._bag = dict()

        if isinstance(seq, bag):
            for e in seq:
                self.add(e, count=seq.get(e))
        else:
            for e in seq:
                self.add(e, count=1)

    def add(self, e, count=1):  
        """
        Add an element to a bag.

        This has no effect if the element is already present.
        """
        if count <= 0:
            pass
        elif e in self._bag:
            self._bag[e] += count
        else:
            self._bag[e] = count

    def clear(self):  
        """ Remove all elements from this bag. """
        self._bag.clear()

    def copy(self):  
        """ Return a shallow copy of a bag. """
        return bag(self)

    def get(self, e):
        return self._bag.get(e, 0)

    def count(self):
        c = 0
        for e in self._bag:
            c += self._bag[e]
        return c

    def difference(self, other):  
        """
        Return the difference of two or more bags as a new bag.

        (i.e. all elements that are in this bag but not the others.)
        """
        diff = bag()
        for e in self._bag:
            diff.add(e, count=self._bag[e])
        for e in other:
            diff.discard(e, count=other.get(e))
        return diff

    def difference_update(self, other):  
        """ Remove all elements of another bag from this bag. """
        for e in other:
            self.discard(e, count=other.get(e))

    def discard(self, e, count=1):  
        """
        Remove an element from a bag if it is a member.

        If the element is not a member, do nothing.
        """
        if e not in self._bag:
            pass
        elif count >= self._bag[e]:
            del self._bag[e]
        else:
            self._bag[e] -= count

    def intersection(self, other):  
        """
        Return the intersection of two bags as a new bag.

        (i.e. all elements that are in both bags.)
        """
        i = bag()
        for e in self._bag:
            if e in other:
                i.add(e, min(self._bag[e], other.get(e)))
        return i

    def intersection_update(self, other):  
        """ Update a bag with the intersection of itself and another. """
        i = self.intersection(other)
        self.clear()
        self.update(i)

    def isdisjoint(self, other):  
        """ Return True if two bags have a null intersection. """
        for e in self._bag:
            if e in other:
                return False
        return True

    def issubbag(self, other):  
        """ Report whether another bag contains this bag. """
        for e in self._bag:
            if e not in other:
                return  False
            if self._bag[e] > other.get(e):
                return False
        return True

    def issuperbag(self, other):  
        """ Report whether this bag contains another bag. """
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
        """
        Remove and return an arbitrary bag element.
        Raises KeyError if the bag is empty.
        """
        for e in self._bag:
            del self._bag[e]
            return e
        raise KeyError("pop from an empty bag")

    def remove(self, e, count=1):  
        """
        Remove an element from a bag; it must be a member.

        If the element is not a member, raise a KeyError.
        """
        if e not in self._bag:
            raise KeyError(e)
        else:
            self.discard(e, count=count)

    def symmetric_difference(self, other):  
        """
        Return the symmetric difference of two bags as a new bag.

        (i.e. all elements that are in exactly one of the bags.)
        """
        sd = bag()
        for e in self._bag:
            sd.add(e, self.get(e) - other.get(e))
        for e in other:
            sd.add(e, other.get(e) - self.get(e))
        return sd

    def symmetric_difference_update(self, other):  
        """ Update a bag with the symmetric difference of itself and another. """
        sdiff = self.symmetric_difference(other)
        self.clear()
        self.update(sdiff)

    def union(self, other):  
        """
        Return the union of bags as a new bag.

        (i.e. all elements that are in either bag.)
        """
        u = bag()
        for e in self._bag:
            u.add(e, count=self.get(e))
        for e in other:
            u.add(e, count=other.get(e))
        return u

    def update(self, other):  
        """ Update a bag with the union of itself and others. """
        for e in other:
            self.add(e, count=other.get(e))

    def __and__(self, other):  
        """ Return self&value. """
        return self.intersection(other)

    def __rand__(self, other):
        """ Return value&self. """
        return self.intersection(other)

    # def __class_getitem__(self, *args, **kwargs):  
    #     """ See PEP 585 """
    #     pass

    def __getitem__(self, e):
        return self._bag[e]

    def __contains__(self, e):
        """ x.__contains__(e) <==> e in x. """
        return e in self._bag

    def __eq__(self, other):  
        """ Return self==value. """
        return self.issamebag(other)

    # def __getattribute__(self, *args, **kwargs):  
    #     """ Return getattr(self, name). """
    #     pass

    def __ge__(self, other):  
        """ Return self>=value. """
        return self.issuperbag(other)

    def __gt__(self, other):  
        """ Return self>value. """
        return self.issuperbag(other) and not self.issamebag(other)

    def __iand__(self, other):  
        """ Return self&=value. """
        self.intersection_update(other)
        return self

    def __ior__(self, other):  
        """ Return self|=value. """
        self.update(other)
        return self

    def __isub__(self, other):  
        """ Return self-=value. """
        self.difference_update(other)
        return self

    def __iter__(self):  
        """ Implement iter(self). """
        return self._bag.__iter__()

    # def __ixor__(self, *args, **kwargs):
    #     """ Return self^=value. """
    #     pass

    def __len__(self, *args, **kwargs):  
        """ Return len(self). """
        return self._bag.__len__()

    def __le__(self, other):  
        """ Return self<=value. """
        return self.issubbag(other)

    def __lt__(self, other):
        """ Return self<value. """
        return self.issubbag(other) and not self.issamebag(other)

    # @staticmethod  # known case of __new__
    # def __new__(*args, **kwargs):  
    #     """ Create and return a new object.  See help(type) for accurate signature. """
    #     pass

    def __ne__(self, other):  
        """ Return self!=value. """
        return not self.issamebag(other)

    def __or__(self, other):  
        """ Return self|value. """
        return self.union(other)

    def __ror__(self, other):
        """ Return value|self. """
        return self.union(other)

    # def __reduce__(self, *args, **kwargs):  
    #     """ Return state information for pickling. """
    #     pass

    def __repr__(self):  
        """ Return repr(self). """
        return self._bag.__repr__()

    def __sub__(self, other):
        """ Return self-value. """
        return self.difference(other)

    def __rsub__(self, other):
        """ Return value-self. """
        return other.difference(self)

    # def __rxor__(self, *args, **kwargs):  
    #     """ Return value^self. """
    #     pass

    # def __sizeof__(self):  ; restored from __doc__
    #     """ S.__sizeof__() -> size of S in memory, in bytes """
    #     pass

    # def __xor__(self, *args, **kwargs):  
    #     """ Return self^value. """
    #     pass

    def __hash__(self):
        return hash(self._bag)
