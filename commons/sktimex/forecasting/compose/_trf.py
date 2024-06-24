from typing import Union, cast

import numpy as np
from sklearn.base import clone

from sktime.forecasting.compose._reduce import _Reducer
from ...transform import LinearTrainTransform, LinearPredictTransform
from ...utils import PD_TYPES


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class ITabularEstimator:
    def fit(self, X, y) -> "ITabularEstimator":
        ...

    def predict(self, X) -> np.ndarray:
        ...


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
#
#   window_length:
#       n           (n,n,F)
#       (n,)        (n,0,F)
#       (,m)        (0,m,F)
#       (n,m)       (n,m,F)
#       (n,m,F)     [[1..n], [1..m]]
#       (n,m,T)     [[1..n], [0..m]]
#
#   ilags:  input lags
#   tlags:  target lags
#

class WindowLength:

    def __init__(self, window_length: Union[int, list, tuple], target=False):
        assert isinstance(window_length, (int, list, tuple))

        # n
        if isinstance(window_length, int):
            target_lags = window_length
            input_lags = None
            current = False
        # [n]
        elif len(window_length) == 1:
            target_lags = window_length[0]
            input_lags = None
            current = False
        # [n, m]
        elif len(window_length) == 2:
            target_lags = window_length[0]
            input_lags = window_length[1]
            current = False
        # [n,m,b]
        elif len(window_length) == 3:
            target_lags = window_length[0]
            input_lags = window_length[1]
            current = bool(window_length[2])
        else:
            raise ValueError(f"Invalid `windows_length`: {window_length}")

        s = 1

        # n : None
        if target_lags is None:
            target_lags = []
        # n : int
        elif isinstance(target_lags, int):
            target_lags = list(range(s, target_lags + s))

        # m : None
        if input_lags is None:
            input_lags = []
        # m, b: int, bool
        elif isinstance(input_lags, int):
            start = 0 if current else 1
            input_lags = list(range(start, input_lags + 1))

        self.input_lags = input_lags
        self.target_lags = target_lags

        # length of the window, that it is different from the number of
        # slots used to create X.
        # For example:
        #
        #       [[1,3], [2,7]]
        #
        # window_length = 7
        #     n_of_lags = 4
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


def _is_fh_seq(fh):
    """ Check if fh is [1,2,...n]"""
    return len(fh) == fh[-1]


def _to_tlags(fh):
    """
    Convert fh into tlags:

        fh = [1,2,3] -> tlags = [0,1,2]

    """
    if fh is None:
        tlags = [0]
    elif fh.is_relative:
        tlags = [int(f-1) for f in fh]
    else:
        raise ValueError("fh is not relative")
    return tlags


# ---------------------------------------------------------------------------
# TabularRegressorForecaster
# ---------------------------------------------------------------------------

class StrategyBasedRegressorForecaster(_Reducer):
    strategy = "any"

    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",
        "requires-fh-in-fit": True,  # is the forecasting horizon required in fit?
        "ignores-exogeneous-X": False,
    }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        windows_identical=True,
        pooling="local",
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
        )
        self.prediction_length = prediction_length
        self.windows_identical = windows_identical

        self._estimators: list[ITabularEstimator] = []
        self.window_length_ = WindowLength(window_length)
        self.prediction_length_ = WindowLength(window_length, True)

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def _transform(self, y, X=None) -> tuple[np.ndarray, np.ndarray]:
        # assert isinstance(y, np.ndarray)
        # assert isinstance(X, (type(None), np.ndarray))

        fh = self.fh
        if not fh.is_relative:
            fh = fh.to_relative(self.cutoff)

        xlags = self.window_length_.xlags
        ylags = self.window_length_.ylags
        tlags = _to_tlags(fh)

        lt = LinearTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags)

        Xt, yt = lt.fit_transform(y=y, X=X)
        return yt, Xt

    def _fit(self,  y, X=None, fh=None):
        assert fh is self._fh

        yt, Xt = self._transform(y, X)

        self.estimators_: list[ITabularEstimator] = []

        if self.strategy in ["direct", "dirrec"]:
            self._fit_multiple(Xt, yt)
        elif self.strategy in ["recursive", "multioutput"]:
            self._fit_single(Xt, yt)
        else:
            raise ValueError(f"Unsupported strategy {self.strategy}")
        return self

    def _fit_single(self, Xt, yt):
        self.estimator.fit(Xt, yt)
        self.estimators_.append(self.estimator)

    def _fit_multiple(self, Xt, yt):
        for i in range(yt.shape[1]):
            yti = yt[:, i]

            estimator: ITabularEstimator = cast(ITabularEstimator, clone(self.estimator))
            estimator.fit(Xt, yti)

            self.estimators_.append(estimator)
    # end

    def _predict(self, fh, X=None):
        # fh MUST BE passed!
        # The check is done by 'sktime'
        self._check_X_y(X=X)
        self._check_fh(fh=fh)

        Xp = X
        yh = self._y
        Xh = self._X

        # n of timeslots to predict
        if not fh.is_relative:
            fh = fh.to_relative(self.cutoff)
        nfh = len(fh)

        xlags = self.window_length_.xlags
        ylags = self.window_length_.ylags
        tlags = _to_tlags(self._fh)

        pt = LinearPredictTransform(xlags=xlags, ylags=ylags, tlags=tlags)
        res = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xp)

        if self.strategy in ["direct", "dirrec"]:
            return self._predict_single(nfh, pt, tlags)
        elif self.strategy in ["recursive", "multioutput"]:
            return self._predict_multiple(nfh, pt)
        else:
            raise ValueError(f"Unsupported strategy {self.strategy}")

    def _predict_multiple(self, nfh, pt):
        y_pred = pt.yp

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            yp: np.ndarray = self.estimator.predict(Xt)

            i = pt.update(i, yp)
        # end
        return y_pred

    def _predict_single(self, nfh, pt, tlags):
        y_pred = pt.yp

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            it = i
            for j, estimator in enumerate(self.estimators_):
                t = tlags[j]

                yp: np.ndarray = estimator.predict(Xt)

                it = pt.update(i, yp, t)

                if it >= nfh: break
            i = it
        # end
        return y_pred

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    def _check_fh(self, fh):
        # if self._fh is not [1,2,...], fh and self._fh must be equals
        if not self.is_fitted:
            return super()._check_fh(fh)
        elif not _is_fh_seq(self._fh):
            return super()._check_fh(fh)
        else:
            return fh

    def _check_X_y(self, X=None, y=None):
        if self._X is not None:
            if X is None:
                raise ValueError(
                    "`X` must be passed to `predict` if `X` is given in `fit`."
                )
            if len(X.shape) != len(self._X.shape):
                raise ValueError(
                    "`X` must be passed to `predict` has a different rank than `X` given in `fit`."
                )
            if X.shape[1] != self._X.shape[1]:
                raise ValueError(
                    "`X` must be passed to `predict` has a different number of columns than `X` given in `fit`."
                )
        if y is not None:
            pass

        return super()._check_X_y(X=X, y=y)

    # def _from_numpy(self, ys: np.ndarray, fh: ForecastingHorizon) -> pd.Series:
    #     ys = ys.reshape(-1)
    #     y_index = fh.to_absolute(self.cutoff)
    #     yp = pd.Series(data=ys, index=y_index.to_pandas())
    #     return yp

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------


class FlexibeDirectRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "direct"


class FlexibleRecursiveRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "recursive"


class FlexibleMultioutputRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "multioutput"


class FlexibleDirRecRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "dirrec"
