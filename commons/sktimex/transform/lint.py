from typing import Optional, cast, Union

import numpy as np
import pandas as pd

from pandasx import to_numpy
from ..utils import is_instance
from sktime.forecasting.base import ForecastingHorizon
from .lagt import LagsTrainTransform, LagsPredictTransform

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _columns(lags: list[int], columns: list[str], predict: bool, forward: bool) -> list[str]:
    names = []
    if forward:
        if not predict:
            for l in lags:
                for c in columns:
                    col = f"{c}@" if l == 1 else f"{c}@{l - 1}"
                    names.append(col)
        else:
            for l in lags:
                for c in columns:
                    col = c if l == 1 else f"{c}@{l-1}"
                    names.append(col)
    else:
        for l in reversed(lags):
            for c in columns:
                col = c if l == 0 else f"{c}_{l}"
                names.append(col)
    return names
# end


def tlags_columns(tlags: list[int], y_columns: list[str]) -> list[str]:
    assert is_instance(tlags, list[int])
    assert is_instance(y_columns, list[str])

    tl_columns = _columns(tlags, y_columns, True, True)
    return tl_columns

def yxulags_columns(
        ylags: list[int],
        xlags: list[int],
        ulags: list[int],
        y_columns: list[str],
        X_columns: Optional[list[str]]) -> list[str]:
    assert is_instance(ylags, list[int])
    assert is_instance(xlags, list[int])
    assert is_instance(ulags, list[int])
    assert is_instance(y_columns, list[str])
    assert is_instance(X_columns, Optional[list[str]])

    yl_columns = _columns(ylags, y_columns, False, False)
    xl_columns = _columns(xlags, X_columns, False, False)
    ul_columns = _columns(ulags, X_columns, False, True)
    return yl_columns + xl_columns + ul_columns
# end


# ---------------------------------------------------------------------------
# LinearTrainTransform
# LinearPredictTransform
# ---------------------------------------------------------------------------

class LinearTrainTransform(LagsTrainTransform):
    def __init__(self, xlags=None, ylags=None, tlags=[1], ulags=[]):
        super().__init__(xlags=xlags, ylags=ylags, tlags=tlags, ulags=ulags, flatten=True)
        self._X_columns = None
        self._y_columns = None
        self._y_index = None

    def predict_transform(self) -> "LinearPredictTransform":
        return LinearPredictTransform(
            xlags=self.xlags, 
            ylags=self.ylags, 
            tlags=self.tlags,
            ulags=self.ulags
        )

    def fit(self, y, X=None):
        self._from_pandas(y, X)
        return super().fit(y=y, X=X)

    def transform(self, *, fh=None, y=None, X=None):
        Xt, yt = super().transform(y=y, X=X, fh=fh)
        Xt, yt = self._to_pandas(Xt, yt)
        return Xt, yt

    # ---------------------------------------------------------------------------

    def _from_pandas(self, y, X):
        if isinstance(y, pd.DataFrame):
            self._y_columns = list(y.columns)
            self._y_index = y.index
        elif isinstance(y, pd.Series):
            self._y_columns = y.name
            self._y_index = y.index
        if isinstance(X, pd.DataFrame):
            self._X_columns = list(X.columns)
        elif isinstance(X, pd.Series):
            raise ValueError("X must be a pd.DataFrame")
        else:
            self._X_columns = []

    def _to_pandas(self, Xt, yt):
        if self._y_index is None:
            return Xt, yt

        n = len(yt)
        y_index = self._y_index[-n:]

        y_columns = [self._y_columns] if isinstance(self._y_columns, str) else self._y_columns
        t_columns = tlags_columns(self.tlags, y_columns)

        if isinstance(self._y_columns, str) and t_columns == [self._y_columns]:
            yt = pd.Series(yt.ravel(), index=y_index, name=self._y_columns)
        else:
            yt = pd.DataFrame(yt, index=y_index, columns=t_columns)
            
        ylags = self.ylags
        xlags = self.xlags if self._X is not None else []
        ulags = self.ulags if self._X is not None else []

        yxu_columns = yxulags_columns(
            ylags, xlags, ulags,
            y_columns, self._X_columns)
        Xt = pd.DataFrame(Xt, index=y_index, columns=yxu_columns)
        return Xt, yt
# end


class LinearPredictTransform(LagsPredictTransform):
    
    def __init__(self, xlags, ylags, tlags, ulags):
        super().__init__(xlags=xlags, ylags=ylags, tlags=tlags, ulags=ulags, flatten=True)

        self._X_columns = None
        self._y_columns = None
        self._yt_index = None
        self._Xt = None
        self._yt = None
        self._cutoff = None

    def fit(self, y, X=None):
        self._from_pandas(y, X)
        self._cutoff = y.index[-1:]
        return super().fit(y=y, X=X)

    def transform(self, *, fh=None, X=None, y=None):
        assert isinstance(fh, ForecastingHorizon)
        Xt, yt = super().transform(fh=len(fh), X=X, y=y)
        self._compose_yt_index(fh)
        Xt = self._x_to_pandas(Xt)
        yt = self._y_to_pandas(yt)
        return Xt, yt

    def _compose_yt_index(self, fh):
        yt_index = fh.to_absolute(self._cutoff).to_pandas()
        self._yt_index = yt_index
        return yt_index

    def step(self, i, t=None) -> pd.DataFrame:
        Xs = self._Xt[i:i+1]
        return Xs

    # ---------------------------------------------------------------------------

    def _from_pandas(self, y, X):
        if isinstance(y, pd.DataFrame):
            self._y_columns = list(y.columns)
        elif isinstance(y, pd.Series):
            self._y_columns = y.name
        if isinstance(X, pd.DataFrame):
            self._X_columns = list(X.columns)
        elif isinstance(X, pd.Series):
            raise ValueError("X must be a pd.DataFrame")
        pass
    # end

    # def _compose_y_index(self, nfh):
    #     # y_index must be inferred based on cutoff and nfh
    #     cutoff = self._cutoff
    #
    #     fh = ForecastingHorizon(list(range(1, nfh+1)), is_relative=True)
    #     fh = fh.to_absolute(cutoff)
    #     y_index = fh.to_pandas()
    #
    #     return y_index

    def _y_to_pandas(self, yt: np.ndarray):
        assert isinstance(yt, np.ndarray)
        assert self._yt_index is not None

        yt_index = self._yt_index

        if isinstance(self._y_columns, str):
            yt = pd.Series(yt.ravel(), index=yt_index, name=self._y_columns)
        else:
            yt = pd.DataFrame(yt, index=yt_index, columns=self._y_columns)

        self._yt = yt

        return yt

    def _x_to_pandas(self, Xt):
        ylags = self.ylags
        xlags = self.xlags if self._Xh is not None else []
        ulags = self.ulags if self._Xh is not None else []

        y_columns = [self._y_columns] if isinstance(self._y_columns, str) else self._y_columns

        y_index = self._yt_index
        yx_columns = yxulags_columns(
            ylags,
            xlags,
            ulags,
            y_columns,
            self._X_columns
        )
        Xt = pd.DataFrame(Xt, index=y_index, columns=yx_columns)

        self._Xt = Xt
        return Xt

    def _x_to_pandas_i(self, Xt, i) -> pd.DataFrame:
        # because in LagsPredictTransform 'Xt' is reused
        # it is not necessary to recreate a dataframe every time.
        # it is enough to reuse it, updating only the index
        # IT DOESN'T WORK!!!

        ylags = self.ylags
        xlags = self.xlags if self._Xh is not None else []
        ulags = self.ulags if self._Xh is not None else []

        y_index = self._yt_index[i:i+1]
        yx_columns = yxulags_columns(
            ylags,
            xlags,
            ulags,
            self._y_columns,
            self._X_columns
        )
        Xt = pd.DataFrame(Xt, index=y_index, columns=yx_columns)
        return Xt

    def predict_steps(self, yf):
        yf = to_numpy(yf, matrix=True)
        Xt, yt = super().predict_steps(yf)
        return self._to_pandas(Xt, yt)

    def _to_pandas(self, Xt, yt):
        # assert self._yt_index is not None
        #
        # y_index = self._yt_index
        #
        # y_columns = tlags_columns([1], self._y_columns)
        # if isinstance(self._y_columns, str):
        #     yt = pd.Series(yt, index=y_index, name=y_columns[0])
        # else:
        #     yt = pd.DataFrame(yt, index=y_index, columns=y_columns)
        #
        # ylags = self.ylags
        # xlags = self.xlags if self._Xh is not None else []
        # ulags = self.ulags if self._Xh is not None else []
        #
        # yxu_columns = yxulags_columns(
        #     ylags, xlags, ulags,
        #     self._y_columns, self._X_columns)
        #
        # Xt = pd.DataFrame(Xt, index=y_index, columns=yxu_columns)
        # return Xt, yt

        Xt = self._Xt
        yt = self._yt
        return Xt, yt
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
