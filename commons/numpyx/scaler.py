import numpy as np


class MinMaxScaler:
    def __init__(self, minr=0, maxr=1):
        self.minr = minr
        self.diffr = maxr - self.minr
        self.minv = 0
        self.diffv = 1
        assert self.diffr != 0

    def fit(self, array: np.ndarray):
        self.minv = array.min()
        self.diffv = array.max() - self.minv
        if abs(self.diffv) < 10e-6:
            self.diffv = 1.
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        data = self.minr + (array - self.minv) / self.diffv * self.diffr
        return data

    def fit_transform(self, array: np.ndarray) -> np.ndarray:
        return self.fit(array).transform(array)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        data = self.minv + (array - self.minr) / self.diffr * self.diffv
        return data
# end
