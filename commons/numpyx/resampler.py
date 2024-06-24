import numpy as np
from .transf import Transformer


def oversample2d(a: np.ndarray, nsamples=10, method='linear') -> np.ndarray:
    """
    Split adjacent values in the mattrix 'a' into 'nsample-1' segments, using a linear interpolation

    :param a' 2D array to analyze
    :param nsamples: number of samples to use for each 2 adjacent values (along the axis 0/rows)
    :param method: method to use (unsupported for now). For default: 'linear'
    """
    assert isinstance(a, np.ndarray) and a.ndim == 2, \
        "Parameter 'a' is not a 2D numpy array"

    n0, m = a.shape
    ns = nsamples * (n0 - 1) + 1
    resampled = np.zeros((ns, m), dtype=a.dtype)

    j = 0
    for i in range(n0-1):
        y0 = a[i + 0]
        y1 = a[i + 1]
        dy = (y1 - y0) / nsamples
        for k in range(nsamples):
            resampled[j] = y0 + k * dy
            j += 1
        # end
    # end
    resampled[-1] = a[-1]

    return resampled
# end


def undersample2d(a: np.ndarray, nsamples=10) -> np.ndarray:
    """
    Invert the oversampling applied with 'oversample2d', selecting a row in 'a'
    each 'nsamples' rows.
    """
    assert isinstance(a, np.ndarray) and len(a.shape) == 2, \
        "Parameter 'a' is not a 2D numpy array"

    #nsamples * (n0 - 1) + 1

    n0, m = a.shape
    ns = (n0 - 1)//nsamples + 1
    resampled = np.zeros((ns, m), dtype=a.dtype)

    j = 0
    for i in range(0, n0, nsamples):
        resampled[j] = a[i]
        j += 1
    resampled[-1] = a[-1]
    return resampled
# end


class Resampler(Transformer):
    """
    Apply the resample rules
    """

    def __init__(self, nsamples=10):
        super().__init__()
        self.nsamples = nsamples

    def fit(self, *data_list: np.ndarray):
        for i, data in enumerate(data_list):
            assert isinstance(data, np.ndarray) and data.ndim == 2, \
                f"Element in position '{i}' is not a 2D numpy array"
            i += 1
        return self

    def transform(self, *data_list: np.ndarray) -> list[np.ndarray]:
        oversampled = []
        for data in data_list:
            oversampled.append(oversample2d(data, self.nsamples))
        # return oversample2d(data, self.nsamples)
        return oversampled

    def inverse_transform(self, *data_list: np.ndarray) -> list[np.ndarray]:
        undersampled = []
        for data in data_list:
            undersampled.append(undersample2d(data, self.nsamples))
        # return undersample2d(data, self.nsamples)
        return undersampled
# end
