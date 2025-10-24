import numpy as np

def identity(n: int, dtype=float) -> np.ndarray:
    m = np.zeros((n,n), dtype=dtype)
    for i in range(n):
        m[i,i] = 1
    return m
# end


def power2(m: np.ndarray, e: int) -> np.ndarray:
    n = len(m)
    p = identity(n, dtype=m.dtype)
    f = m
    for i in range(e):
        p = np.dot(p, f)
        f = np.dot(f, f)
    return p
# end