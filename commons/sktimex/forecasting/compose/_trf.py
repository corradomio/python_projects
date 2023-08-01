from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import clone
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose._reduce import _Reducer

from numpyx import LinearTrainTransform, LinearPredictTransform


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class ITabularEstimator:
    def fit(self, X, y):
        ...

    def predict(self, X):
        ...


class WindowLength:

    def __init__(self, window_length: Union[int, list, tuple]):
        assert isinstance(window_length, (int, list, tuple))

        if isinstance(window_length, int):
            target_lags = window_length
            input_lags = None
            current = False
        elif len(window_length) == 1:
            target_lags = window_length[0]
            input_lags = None
            current = False
        elif len(window_length) == 2:
            target_lags = window_length[0]
            input_lags = window_length[1]
            current = False
        elif len(window_length) == 3:
            target_lags = window_length[0]
            input_lags = window_length[1]
            current = bool(window_length[2])
        else:
            raise ValueError(f"Invalid `windows_length`: {window_length}")

        if isinstance(target_lags, int):
            target_lags = list(range(1, target_lags + 1))
        elif target_lags is None:
            target_lags = []
        if isinstance(input_lags, int):
            start = 0 if current else 1
            input_lags = list(range(start, input_lags + 1))
        elif input_lags is None:
            input_lags = target_lags

        self.input_lags = input_lags
        self.target_lags = target_lags
        self.window_length = max(
            max(self.target_lags) if len(self.target_lags) > 0 else 0,
            max(self.input_lags) if len(self.input_lags) > 0 else 0
        )
    # end

    def __len__(self):
        return self.window_length

    @property
    def xlags(self):
        return self.input_lags

    @property
    def ylags(self):
        return self.target_lags
# end


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise ValueError(f"Not a np.ndarray compatible typ")
# end


# ---------------------------------------------------------------------------
# TabularRegressorForecaster
# ---------------------------------------------------------------------------

class TabularRegressorForecaster(_Reducer):
    strategy = "any"

    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
        "ignores-exogeneous-X": False,
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        pooling="local",
    ):
        super().__init__(estimator, window_length, transformers, pooling)
        self._estimators: list[ITabularEstimator] = []
        self.window_length_ = WindowLength(window_length)

    def get_params(self, deep=True):
        params = {}
        params['estimator'] = self.estimator
        params['window_length'] = self.window_length
        return params
    # end

    def fit(self, y, X=None, fh=None):
        if fh is None:
            fh = [1]
        return super().fit(y=y, X=X, fh=fh)

    def _transform(self, y, X=None) -> tuple[np.ndarray, np.ndarray]:
        X = _to_numpy(X)
        y = _to_numpy(y)

        fh = self.fh

        xlags = self.window_length_.xlags
        ylags = self.window_length_.ylags
        tlags = list(fh - 1)
        lt = LinearTrainTransform(xlags, ylags, tlags)

        Xt, yt = lt.fit_transform(X, y)
        return yt, Xt

    def _fit(self, y, X=None, fh=None):
        assert fh is self._fh and fh is self.fh

        yt, Xt = self._transform(y, X)

        self.estimators_ = []

        for i in range(yt.shape[1]):
            yti = yt[:, i]

            estimator: ITabularEstimator = clone(self.estimator)
            estimator.fit(Xt, yti)

            self.estimators_.append(estimator)
        return self

    def _predict(self, fh, X=None):
        fh = fh.to_relative(self.cutoff)

        Xp = _to_numpy(X)
        yh = _to_numpy(self._y)
        Xh = _to_numpy(self._X)

        # X, yh, Xh
        n = int(fh[-1])

        xlags = self.window_length_.xlags
        ylags = self.window_length_.ylags

        pt = LinearPredictTransform(xlags=xlags, ylags=ylags)
        y_pred = pt.fit(X=Xh, y=yh).transform(X=Xp, fh=n)

        i = 0
        while i < n:
            Xt = pt.step(i)
            for j, estimator in enumerate(self.estimators_):
                if i+j >= n: continue
                yp: np.ndarray = estimator.predict(Xt)
                y_pred[i] = yp[0]
                i += 1
        # end
        # add the index
        y_pred = self._from_numpy(y_pred, fh)
        return y_pred

    def _from_numpy(self, ys: np.ndarray, fh: ForecastingHorizon) -> pd.Series:
        ys = ys.reshape(-1)
        y_index = fh.to_absolute(self.cutoff)
        yp = pd.Series(data=ys, index=y_index)
        return yp
# end
