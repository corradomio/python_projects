import numpy as np

def const_wave(x: np.ndarray, c: float = 0) -> tuple[np.ndarray, np.ndarray]:
    y = np.zeros(x.shape, dtype=float) + c
    return y

def sin_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    y = np.sin(2*np.pi*x)
    return y

def sinabs_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    y = np.abs(np.sin(2*np.pi*x))
    return y

def cos_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    y = np.cos(2*np.pi*x)
    return y

def sawtooth_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    y = np.mod(x, 1)
    return y

def inverted_sawtooth_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    o = np.ones_like(x, dtype=float)
    y = o-np.mod(x, 1)
    return y

def square_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    s = np.mod(x, 1)
    y = np.ones_like(x, dtype=float)
    y[s > 0.5] = -1
    return y

def triangle_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    s = np.mod(x, 1)
    y = np.ones_like(x, dtype=float)
    y[s > 0.5] = 4*(1 - s[s > 0.5]) - 1.
    y[s < 0.5] = 4*(    s[s < 0.5]) - 1.
    return y

def sintooth_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    y = np.abs(np.sin(np.pi*x))
    return y

def triangletooth_wave(x: np.ndarray, offset=0) -> np.ndarray:
    x = x.astype(float) + offset
    s = np.mod(x, 1)
    y = np.ones_like(x, dtype=float)
    y[s > 0.5] = 2 * (1 - s[s > 0.5])
    y[s < 0.5] = 2 * (    s[s < 0.5])
    return y

# ---------------------------------------------------------------------------
# fourier_wave
# ---------------------------------------------------------------------------

def fourier_wave(
    c: float,
    a: np.ndarray|list[float],
    p: np.ndarray|list[float],
    x: np.ndarray,
    offset: float=0
) -> np.ndarray:
    if len(p) < len(a):
        p = list(p) + [0]*(len(a)-len(p))
    assert len(a) == len(p)
    n = len(a)
    x = x.astype(float) + offset
    y = c*np.ones_like(x, dtype=float)

    for i in range(n):
        y += a[i]*sin_wave((i+1)*x, p[i])
    return y

# ---------------------------------------------------------------------------
# hadamard_wave
# ---------------------------------------------------------------------------

def hadamard_wave(
    c: float,
    a: np.ndarray|list[float],
    p: np.ndarray|list[float],
    x: np.ndarray,
    offset: float=0
) -> np.ndarray:
    if len(p) < len(a):
        p: list[float] = list(p) + [0]*(len(a)-len(p))
    assert len(a) == len(p)
    n = len(a)
    x = x.astype(float) + offset
    y = c*np.ones_like(x, dtype=float)

    for i in range(n):
        y += a[i]*square_wave((i+1)*x, p[i])
    return y

# ---------------------------------------------------------------------------
# slant_haar_wave
# ---------------------------------------------------------------------------

def slant_haar_wave(
    c: float,
    a: np.ndarray | list[float],
    p: np.ndarray | list[float],
    x: np.ndarray,
    offset: float=0
) -> np.ndarray:
    if len(p) < len(a):
        p: list[float] = list(p) + [0]*(len(a)-len(p))
    assert len(a) == len(p)
    n = len(a)
    x = x.astype(float) + offset
    y = c*np.ones_like(x, dtype=float)

    for i in range(n):
        y += a[i]*triangle_wave((i+1)*x, p[i])
    return y


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# "_xy": (y, x) with in y all positive values
#
