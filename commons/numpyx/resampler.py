import numpy as np


def oversample2d(a: np.ndarray, nsamples=10) -> np.ndarray:
    """

    """
    assert isinstance(a, np.ndarray) and len(a.shape) == 2, \
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


class Resampler:

    def __init__(self, nsamples=10):
        self.nsamples = nsamples

    def fit(self, *data_list: np.ndarray):
        for i, data in enumerate(data_list):
            assert isinstance(data, np.ndarray) and len(data.shape) == 2, \
                f"Element in position '{i}' is not a 2D numpy array"
            i += 1
        return self

    def transform(self, *data_list: np.ndarray) -> list[np.ndarray]:
        oversampled = []
        for data in data_list:
            oversampled.append(oversample2d(data, self.nsamples))
        # return oversample2d(data, self.nsamples)
        return oversampled


    def fit_transform(self, *data: np.ndarray):
        return self.fit(*data).transform(*data)

    def inverse_transform(self, *data_list: np.ndarray) -> list[np.ndarray]:
        undersampled = []
        for data in data_list:
            undersampled.append(undersample2d(data, self.nsamples))
        # return undersample2d(data, self.nsamples)
        return undersampled
# end
