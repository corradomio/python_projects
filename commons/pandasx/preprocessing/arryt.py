import numpy as np
from pandas import DataFrame

from .base import BaseEncoder
from .lagt import _resolve_xylags


class ArrayTransformer(BaseEncoder):
    def __init__(self, columns=None, target=None, xlags=None, ylags=None, dtype=np.float32, swap=False):
        """

        :param columns: columns used as target
        :param lags: lags used for the features
        :param tlags: lags used for the targets
        """
        super().__init__(target if target else columns, False)
        self.xlags = xlags
        self.ylags = ylags
        self.dtype = dtype
        self.swap = swap

        self._xlags = _resolve_xylags(xlags)
        self._ylags = _resolve_xylags(ylags, True)

    def transform(self, df: DataFrame):
        dtype = self.dtype
        targets = self.columns
        swap = self.swap

        X = df.to_numpy(dtype=dtype)
        y = df[targets].to_numpy(dtype=dtype)

        xlags = list(reversed(self._xlags))
        ylags = self._ylags

        n = len(X)          # n of rows
        nx = len(xlags)     # n of lags
        ny = len(ylags)     # n of lags
        sx = max(xlags)     # window length
        sy = max(ylags)     # window length
        mx = X.shape[1]     # n of features
        my = y.shape[1]     # n of features
        nt = n - (sx + sy)  # n of predicted rows

        if swap:
            Xt = np.zeros((nt, mx, nx))
            yt = np.zeros((nt, my, ny))
        else:
            Xt = np.zeros((nt, nx, mx))
            yt = np.zeros((nt, ny, my))

        for i in range(nx):
            k = xlags[i]
            ik = sx - k
            if swap:
                Xt[:, :, i] = X[ik:ik + nt]
            else:
                Xt[:, i, :] = X[ik:ik + nt]

        for i in range(ny):
            k = ylags[i]
            ik = sx + k    # skip the X window
            if swap:
                yt[:, :, i] = y[ik:ik + nt]
            else:
                yt[:, i, :] = y[ik:ik + nt]

        return Xt, yt
# end



