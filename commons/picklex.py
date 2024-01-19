from typing import Any
import pickle


def save(obj: Any, file: str):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load(file: str) -> Any:
    with open(file, 'rb') as f:
        return pickle.load(f)
