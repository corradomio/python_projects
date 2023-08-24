class bag(object):
    """
    bag() -> new empty bag object
    bag(iterable) -> new bag object

    Build an unordered collection of unique elements.
    """

    def __init__(self, seq=()):
        self._bag = dict()

        if isinstance(seq, dict):
            for e in seq:
                self.add(e, count=seq[e])
        else:
            for e in seq:
                self.add(e, count=1)

    def add(self, e, count=1):  # real signature unknown
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


    # def clear(self, *args, **kwargs):  # real signature unknown
    #     """ Remove all elements from this bag. """
    #     pass

    def copy(self):  # real signature unknown
        """ Return a shallow copy of a bag. """
        return bag(self)

    def get(self, e, defval=0):
        return self._bag.get(e, defval)

    def count(self, e=None):
        if e is not None:
            return  self._bag.get(e, 0)
        c = 0
        for e in self._bag:
            c += self.count(e)
        return c

    def difference(self, other):  # real signature unknown
        """
        Return the difference of two or more bags as a new bag.

        (i.e. all elements that are in this bag but not the others.)
        """
        diff = bag()
        for e in self._bag:
            diff.add(e, count=self.count(e))
        for e in other:
            diff.discard(e, count=other.count(e))
        return diff

    def difference_update(self, other):  # real signature unknown
        """ Remove all elements of another bag from this bag. """
        for e in other:
            self.discard(e, count=other.count(e))

    def discard(self, e, count=1):  # real signature unknown
        """
        Remove an element from a bag if it is a member.

        If the element is not a member, do nothing.
        """
        if e not in self._bag:
            pass
        elif count >= self.count(e):
            del self._bag[e]
        else:
            self._bag[e] = self.count(e) - count

    def intersection(self, other):  # real signature unknown
        """
        Return the intersection of two bags as a new bag.

        (i.e. all elements that are in both bags.)
        """
        i = bag()
        for e in self._bag:
            count = min(self.count(e), other.get(e, 0))
            i.add(e, count)
        return i

    def intersection_update(self, other):  # real signature unknown
        """ Update a bag with the intersection of itself and another. """
        for e in other:
            self.discard(e, other.count(e))

    def isdisjoint(self, other):  # real signature unknown
        """ Return True if two bags have a null intersection. """
        for e in self._bag:
            if e in other:
                return False
        return  True

    def issubbag(self, other):  # real signature unknown
        """ Report whether another bag contains this bag. """
        for e in self._bag:
            if e not in other:
                return  False
            if self.count(e) > other.count(e):
                return False
        return True

    def issuperbag(self, other):  # real signature unknown
        """ Report whether this bag contains another bag. """
        for e in self._bag:
            if e not in other:
                return False
            if self.count(e) < other.count(e):
                return False
        return True

    def issamebag(self, other):
        for e in self._bag:
            if e not in other:
                return False
            if self.count(e) != other.count(e):
                return False
        return True

    def pop(self):  # real signature unknown
        """
        Remove and return an arbitrary bag element.
        Raises KeyError if the bag is empty.
        """
        for e in self._bag:
            del self._bag[e]
            return e
        raise KeyError("empty")

    def remove(self, e, count=1):  # real signature unknown
        """
        Remove an element from a bag; it must be a member.

        If the element is not a member, raise a KeyError.
        """
        if e not in self._bag:
            raise KeyError(e)
        self.discard(e, count=count)

    def symmetric_difference(self, other):  # real signature unknown
        """
        Return the symmetric difference of two bags as a new bag.

        (i.e. all elements that are in exactly one of the bags.)
        """
        sd = bag()
        for e in self._bag:
            sd.add(e, self.count(e)-other.get(e, 0))
        for e in other:
            sd.add(e, other.count(e) - self.get(e, 0))
        return sd

    def symmetric_difference_update(self, other):  # real signature unknown
        """ Update a bag with the symmetric difference of itself and another. """
        sdiff = self.symmetric_difference(other)
        self.clear()
        self.union(sdiff)

    def union(self, other):  # real signature unknown
        """
        Return the union of bags as a new bag.

        (i.e. all elements that are in either bag.)
        """
        u = bag()
        for e in self._bag:
            u.add(e, count=self.count(e))

        if isinstance(other, bag):
            for e in other:
                u.add(e, count=other.count(e))
        else:
            for e in other:
                u.add(e)
        return u

    def update(self, other):  # real signature unknown
        """ Update a bag with the union of itself and others. """
        if isinstance(other, bag):
            for e in other:
                self.add(e, count=other.count(e))
        else:
            for e in other:
                self.add(e)

    def __and__(self, other):  # real signature unknown
        """ Return self&value. """
        return self.intersection(other)

    # def __class_getitem__(self, *args, **kwargs):  # real signature unknown
    #     """ See PEP 585 """
    #     pass

    def __getitem__(self, e):
        return self._bag[e]

    def __contains__(self, e):  # real signature unknown; restored from __doc__
        """ x.__contains__(e) <==> e in x. """
        return e in self._bag

    def __eq__(self, other):  # real signature unknown
        """ Return self==value. """
        return self.issamebag(other)

    # def __getattribute__(self, *args, **kwargs):  # real signature unknown
    #     """ Return getattr(self, name). """
    #     pass

    def __ge__(self, other):  # real signature unknown
        """ Return self>=value. """
        return self.issuperbag(other)

    def __gt__(self, other):  # real signature unknown
        """ Return self>value. """
        return self.issuperbag(other) and not self.issamebag(other)

    def __iand__(self, other):  # real signature unknown
        """ Return self&=value. """
        self.intersection_update(other)

    # def __init__(self, seq=()):  # known special case of bag.__init__
    #     """
    #     bag() -> new empty bag object
    #     bag(iterable) -> new bag object
    #
    #     Build an unordered collection of unique elements.
    #     # (copied from class doc)
    #     """
    #     pass

    def __ior__(self, other):  # real signature unknown
        """ Return self|=value. """
        self.update(other)

    def __isub__(self, other):  # real signature unknown
        """ Return self-=value. """
        self.difference_update(other)

    def __iter__(self):  # real signature unknown
        """ Implement iter(self). """
        return self._bag.__iter__()

    def __ixor__(self, *args, **kwargs):  # real signature unknown
        """ Return self^=value. """
        pass

    def __len__(self, *args, **kwargs):  # real signature unknown
        """ Return len(self). """
        return self._bag.__len__()

    def __le__(self, other):  # real signature unknown
        """ Return self<=value. """
        return self.issubbag(other)

    def __lt__(self, *args, **kwargs):  # real signature unknown
        """ Return self<value. """
        return self.issubbag(other) and not self.issamebag(other)

    # @staticmethod  # known case of __new__
    # def __new__(*args, **kwargs):  # real signature unknown
    #     """ Create and return a new object.  See help(type) for accurate signature. """
    #     pass

    def __ne__(self, other):  # real signature unknown
        """ Return self!=value. """
        return not self.issamebag(other)

    def __or__(self, other):  # real signature unknown
        """ Return self|value. """
        return self.union(other)

    def __rand__(self, other):  # real signature unknown
        """ Return value&self. """
        return self.intersection(other)

    # def __reduce__(self, *args, **kwargs):  # real signature unknown
    #     """ Return state information for pickling. """
    #     pass

    def __repr__(self):  # real signature unknown
        """ Return repr(self). """
        return self._bag.__repr__()

    def __ror__(self, other):  # real signature unknown
        """ Return value|self. """
        return self.union(other)

    def __rsub__(self, other):  # real signature unknown
        """ Return value-self. """
        return  other.difference(self)

    # def __rxor__(self, *args, **kwargs):  # real signature unknown
    #     """ Return value^self. """
    #     pass

    # def __sizeof__(self):  # real signature unknown; restored from __doc__
    #     """ S.__sizeof__() -> size of S in memory, in bytes """
    #     pass

    def __sub__(self, other):  # real signature unknown
        """ Return self-value. """
        return self.difference(other)

    # def __xor__(self, *args, **kwargs):  # real signature unknown
    #     """ Return self^value. """
    #     pass

    __hash__ = None
