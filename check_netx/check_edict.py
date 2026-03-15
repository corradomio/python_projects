# class dict(object):
#     def clear(self):  # real signature unknown; restored from __doc__
#     def copy(self):  # real signature unknown; restored from __doc__
#
#     def get(self, *args, **kwargs):  # real signature unknown
#
#     def items(self):  # real signature unknown; restored from __doc__
#     def keys(self):  # real signature unknown; restored from __doc__
#     def pop(self, k, d=None):  # real signature unknown; restored from __doc__
#     def popitem(self, *args, **kwargs):  # real signature unknown
#     def update(self, E=None, **F):  # known special case of dict.update
#     def values(self):  # real signature unknown; restored from __doc__
#
#     def __contains__(self, *args, **kwargs):  # real signature unknown
#
#     def __getitem__(self, *args, **kwargs):  # real signature unknown
#     def __setitem__(self, *args, **kwargs):  # real signature unknown
#     def __delitem__(self, *args, **kwargs):  # real signature unknown
#
#     def __eq__(self, *args, **kwargs):  # real signature unknown
#     def __getattribute__(self, *args, **kwargs):  # real signature unknown
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
#     def __sizeof__(self):  # real signature unknown; restored from __doc__
#     __hash__ = None


def ukey(key):
    u, v = key
    return key if u < v else (v, u)


class udict(dict):
    def __init__(self):
        super().__init__()

    def get(self, key, default=None):
        return super().get(ukey(key), default)

    def __getitem__(self, key):
        return super().__getitem__(ukey(key))

    def __setitem__(self, key, value):
        return super().__setitem__(ukey(key), value)

    def __contains__(self, key):
        return super().__contains__(ukey(key))

    def __delitem__(self, key):
        return super().__delitem__(ukey(key))



uedges = udict()

uedges[(2,1)] = {"w": 1}
print(uedges[(1,2)])
print(uedges[(2,1)])
print((1,2) in uedges)
print((2,1) in uedges)
del uedges[(2,1)]