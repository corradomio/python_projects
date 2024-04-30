#
# Simple cache based on dictionary
#
from datetime import datetime


def now():
    return int(datetime.timestamp(datetime.now()))


class DataCache:
    def __init__(self, timeout=3):
        self.timeout = timeout
        self._cache = dict()
        self._tinfo = dict()    # time info: (timestamp, timeout)

    def get(self, key, defval=None):
        if key not in self._cache:
            return defval
        timestamp, timeout = self._tinfo[key]
        if (now() - timestamp) > timeout:
            del self._cache[key]
            del self._tinfo[key]
            return defval
        else:
            return self._cache[key]

    def set(self, key, value, timeout=0):
        self._cache[key] = value
        self._tinfo[key] = (now(), timeout if timeout > 0 else self.timeout)

    def clear(self):
        self._cache.clear()
        self._tinfo.clear()
# end
