from typing import Optional


# ---------------------------------------------------------------------------
# InMemoryCache
# ---------------------------------------------------------------------------

class CacheManager:
    def cache(self, name: str, ktype: Optional[type], vtype: Optional[type]):
        pass


class InMemoryCache(dict, CacheManager):

    def __init__(self, manager, name: str, ktype: Optional[type] = None, vtype: Optional[type] = None):
        super().__init__()
        self.manager = manager
        self.name = name
        self.ktype = ktype
        self.vtype = vtype
        self.dict = dict()

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, key):
        return self.dict.__contains__(key)

    def __len__(self):
        return self.dict.__len__()

    def __iter__(self):
        return self.dict.__iter__()


class InMemoryCaches(InMemoryCache):

    @staticmethod
    def open(name: str = "", ktype: Optional[type] = None, vtype: Optional[type] = None):
        return InMemoryCaches(name, ktype, vtype)

    def __init__(self, name: str, ktype: Optional[type], vtype: Optional[type]):
        super().__init__(self, name, ktype, vtype)
        self.caches: dict[str, InMemoryCache] = dict()
        self.caches[name] = self

    def close(self):
        pass

    def cache(self, name: str, ktype: Optional[type], vtype: Optional[type]):
        if name in self.caches:
            raise ValueError(f"Cache {name} already defined")
        cache = InMemoryCache(self, name, ktype, vtype)
        self.caches[name] = cache
        return cache

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
# end


# ---------------------------------------------------------------------------
# Encoders/decoders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PersistentCache
# ---------------------------------------------------------------------------

class PersistenCache(dict):

    def __init__(self, manager, name: str, ktype: Optional[type], vtype: Optional[type]):
        super().__init__()
        self.manager = manager
        self.name = name
        self.ktype = ktype
        self.vtype = vtype
        self.dict = dict()