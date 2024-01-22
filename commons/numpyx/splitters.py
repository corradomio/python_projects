import numpy as np
from typing import Union


def train_test_split(*arrys, train_size:Union[int,float]=0, test_size:Union[int,float]=0):
    """
    Split the arrays in 'arrys' in train/test.
    The parameter 'train_size' is alternative to 'test_size'
    The parameters 'train_size', 'test_size' can be specified as integer values or
    float values less than 1.

    """
    def _normalize_sizes(n, train_size, test_size):
        if 0 < train_size < 1:
            train_size = int(n * train_size)
        if 0 < test_size < 1:
            test_size = int(n * test_size)
        if test_size > 0:
            train_size = n - test_size
        return train_size

    splits = []
    for a in arrys:
        train_size = _normalize_sizes(len(a), train_size, test_size)
        splits.append(a[:train_size])
        splits.append(a[train_size:])
    return splits
# end


def size_split(*arrys, size:Union[int,float]=.5) -> list[np.ndarray]:
    return train_test_split(*arrys, train_size=size)
