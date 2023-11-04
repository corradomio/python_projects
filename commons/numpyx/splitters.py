import numpy as np


def size_split(*arrys, split_size=.5) -> list[np.ndarray]:
    def sz(a):
        if split_size > 1:
            return split_size
        else:
            return int(split_size*len(a))

    splits = []
    for a in arrys:
        n = sz(a)
        splits.append(a[:n])
        splits.append(a[n:])
    return splits
# end