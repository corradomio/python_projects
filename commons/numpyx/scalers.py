import numpy as np


# class MinMaxScaler:
#     def __init__(self, minr=0, maxr=1):
#         self.minr = minr
#         self.diffr = maxr - self.minr
#         self.minv = 0
#         self.diffv = 1
#         assert self.diffr != 0
#
#     def fit(self, array: np.ndarray):
#         self.minv = array.min()
#         self.diffv = array.max() - self.minv
#         if abs(self.diffv) < 10e-6:
#             self.diffv = 1.
#         return self
#
#     def transform(self, array: np.ndarray) -> np.ndarray:
#         data = self.minr + (array - self.minv) / self.diffv * self.diffr
#         return data
#
#     def fit_transform(self, array: np.ndarray) -> np.ndarray:
#         return self.fit(array).transform(array)
#
#     def inverse_transform(self, array: np.ndarray) -> np.ndarray:
#         data = self.minv + (array - self.minr) / self.diffr * self.diffv
#         return data
# # end

class Scaler:

    def fit(self, array) -> "Scaler":
        ...

    def transform(self, array: np.ndarray) -> np.ndarray:
        ...

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, array):
        return self.fit(array).transform(array)


class MinMaxScaler(Scaler):
    """
    Apply a scaling to the array in such way that all valuer are in
    the range [min, max].

    It is possible to apply the scaling globally or per column
    """

    def __init__(self, min=0., max=1., range=None, globally=False):
        """
        Initialize the transformer.

        Default range: [0, 1]

        :param min: min range value
        :param max: max range value
        :param globally: if to apply the transformation globally or by column
        """
        super().__init__()

        if range is not None:
            min, max = range

        self.minr = min
        self.diffr = max - min

        self.minv = None
        self.diffv = None

        self.globally = globally
        self.rank = 0
        assert self.diffr != 0

    def fit(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)

        rank = len(array.shape)
        globally = self.globally
        if globally or rank == 1:
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

        if self.globally or self.rank == 1:
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

        if self.globally or self.rank == 1:
            scaled = self.minv + (array - self.minr) / self.diffr * self.diffv
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.minv[i] + (array[:, i] - self.minr) / self.diffr * self.diffv[i]
        return scaled
# end


class NormalScaler(Scaler):
    """
   Apply a scaling to the array in such way that all valuer have
   a normal distribution with parameters meanr and stdr

   It is possible to apply the scaling globally or per column
   """

    def __init__(self, mean=0., std=1., params=None, clip=None, globally=False):
        """
        Initialize the transformer

        Default distribution parameters: mean=0, std=1

        :param mean: mean
        :param std: standard deviation
        :param globally: if to apply the transformation globally or by column
        """
        super().__init__()

        if params is not None:
            mean, std = params

        self.meanr = mean
        self.stdr = std
        self.clip = clip

        self.meanv = None
        self.stdv = None

        self.globally = globally
        self.rank = 0

    def fit(self, array: np.ndarray):
        assert isinstance(array, np.ndarray)

        rank = len(array.shape)
        globally = self.globally
        if globally or rank == 1:
            self.meanv = array.mean()
            self.stdv = array.std()
        elif rank == 2:
            self.meanv = array.mean(axis=0)
            self.stdv = array.std(axis=0)
        else:
            raise ValueError(f"Unsupported array with rank={rank}")

        self.rank = rank
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)
        assert self.rank == len(array.shape)

        if self.globally or self.rank == 1:
            scaled = self.meanr + (array - self.meanv) / self.stdv * self.stdr
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.meanr + (array[:, i] - self.meanv[i]) / self.stdv[i] * self.stdr
        # end
        if self.clip is not None:
            minc, maxc = self.clip
            scaled[scaled < minc] = minc
            scaled[scaled > maxc] = maxc
        return scaled

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        assert isinstance(array, np.ndarray)
        assert self.rank == len(array.shape)

        if self.globally or self.rank == 1:
            scaled = self.meanv + (array - self.meanr) / self.stdr * self.stdv
        else:
            scaled = np.zeros_like(array)
            for i in range(array.shape[1]):
                scaled[:, i] = self.meanv[i] + (array[:, i] - self.meanr) / self.stdr * self.stdv[i]
        return scaled
# end

