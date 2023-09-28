import numpy as np


class Transformer:

    def fit(self, array) -> "Transformer":
        ...

    def transform(self, array: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, array):
        return self.fit(array).transform(array)


class MinMaxScaler(Transformer):
    """
    Apply a global scaling on the entire array
    """

    def __init__(self, minr=0, maxr=1, bycolumns=False):
        super().__init__()

        self.minr = minr
        self.diffr = maxr - self.minr

        self.minv = None
        self.diffv = None

        self.cols = bycolumns
        self.rank = 0
        assert self.diffr != 0

    def fit(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)

        rank = len(array.shape)
        cols = self.cols
        if not cols or rank == 1:
            self.minv = array.min()
            self.diffv = array.max() - self.minv
            if abs(self.diffv) < 10e-6:
                self.diffv = 1.
        elif rank == 2:
            self.minv = array.min(axis=0)
            self.diffv = array.max(axis=0) - self.minv
        else:
            raise ValueError(f"Unsupported array with rank={rank}")

        self.rank = rank
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)
        assert self.rank == len(array.shape)

        if not self.cols or self.rank == 1:
            scaled = self.minr + (array - self.minv) / self.diffv * self.diffr
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.minr + (array[:, i] - self.minv[i]) / self.diffv[i] * self.diffr
        # end
        return scaled

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)
        assert self.rank == len(array.shape)

        if not self.cols or self.rank == 1:
            scaled = self.minv + (array - self.minr) / self.diffr * self.diffv
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.minv[i] + (array[:, i] - self.minr) / self.diffr * self.diffv[i]
        return scaled
# end
