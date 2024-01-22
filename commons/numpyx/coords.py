import numpy as np

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

