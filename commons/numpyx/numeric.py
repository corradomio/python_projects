import numpy as np


def argsort(a: np.ndarray, axis=-1, kind=None, order=None, desc=False) -> np.ndarray:
    """
    Extends np.argsort with 'desc' (descendent) parameter
    """
    s = np.argsort(a, axis=axis, kind=kind, order=order)
    return np.flip(s) if desc else s


def chop(*alist, eps=1e-8):
    """
    Clip all values in range [-eps, +eps] to zero
    """
    def _chop(a):
        if isinstance(a, list):
            return list(chop(e) for e in a)
        elif isinstance(a, tuple):
            return tuple(chop(e) for e in a)
        elif a.dtype not in [np.complex64, np.complex128]:
            c = a.copy()
            c[np.abs(c) <= eps] = 0.
        else:
            r = chop(np.real(a))
            i = chop(np.imag(a))
            c = np.vectorize(complex)(r, i)
        return c

    if len(alist) == 1:
        return _chop(alist[0])
    else:
        return [_chop(a) for a in alist]


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
