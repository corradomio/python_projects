import numpy as np


def argsort(a: np.ndarray, axis=-1, kind=None, order=None, desc=False) -> np.ndarray:
    """
    Extends np.argsort with 'desc' (descendent) parameter
    """
    s = np.argsort(a, axis=axis, kind=kind, order=order)
    return np.flip(s) if desc else s


def chop(*alist, eps=1e-8):
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
# Complex
# ---------------------------------------------------------------------------

def to_rect(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.real(a), np.imag(a)


def from_rect(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    return re + im*1j


def to_polar(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(a), np.angle(a)


def from_polar(ro, phi):
    return ro*(np.cos(phi) + np.sin(phi)*1j)


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
