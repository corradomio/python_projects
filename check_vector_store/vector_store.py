import os.path
import time
import numpy as np

#
# key -> nparray OR dict[str, nparray] ???
#


def _timestamp() -> int:
    return int(time.time()*1000)


class _Entry:
    def __init__(self, key, value):
        self.key: str = key
        self.value: np.ndarray = value
        self.timestamp: int = _timestamp()

class VectorStore:

    def __init__(self, max_keys: int=128, store_folder: str=".vector_store", check_time: int=5000):
        self._max_keys: int = max_keys
        self._store_folder: str = store_folder
        self._store: dict[str, _Entry] = {}
        self._last_check: int = 0
        self._check_time: int = check_time

        if not os.path.exists(self._store_folder):
            os.makedirs(self._store_folder, exist_ok=True)

    def put(self, key: str, embedding: np.ndarray):
        self._store[key] = _Entry(key, embedding)
        self._check()

    def get(self, key: str) -> np.ndarray:
        if key in self._store:
            e = self._store[key]
            e.timestamp = _timestamp()
            return e.value
        key_path = f"{self._store_folder}/{key}"
        if not os.path.exists(key_path):
            raise ValueError(f"Key {key} not existent")
        embedding = np.load(key_path)
        self._store[key] = _Entry(key, embedding)
        return embedding

    def _check(self):
        if _timestamp() - self._last_check < self._check_time:
            return
        if len(self._store) < self._max_keys:
            return

        entries = self._store.values()
        entries: list[_Entry] = sorted(entries, key=lambda e: e.timestamp)
        for e in entries[:-self._max_keys]:
            key_path = f"{self._store_folder}/{e.key}"
            if not os.path.exists(key_path):
                # save array
                np.save(key_path, e.value)
            del self._store[e.key]
        self._last_check = _timestamp()
# end



