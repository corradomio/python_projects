from typing import Union, cast

import numpy as np
from sklearn.base import clone
from sktime.forecasting.base import ForecastingHorizon

from sktime.forecasting.compose._reduce import _Reducer
from ...transform import LinearTrainTransform, LinearPredictTransform
from ...transform._utils import lmax


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
#   window_length -> (y_lags, x_lags):
#       n           (n,n)
#       (n,)        (n,0)
#       (,m)        (0,m)
#       (n,m)       (n,m)
#

class WindowLength:

    def __init__(self, window_length: Union[int, list, tuple], is_tlags):
        assert isinstance(window_length, (int, list, tuple))
        # length of the window, that it is different from the number of
        # slots used to create X.
        # For example:
        #
        #       [[1,3], [2,7]]
        #
        # window_length = 7
        #     n_of_lags = 4

        if is_tlags:
            self._compose_tlags(window_length)
        else:
            self._compose_xylags(window_length)
    # end

    def _compose_xylags(self, window_length: Union[int, list, tuple]):
        assert isinstance(window_length, (int, list, tuple))

        # n
        if isinstance(window_length, int):
            ylags = window_length
            xlags = window_length
        # [n]
        elif len(window_length) == 1:
            ylags = window_length[0]
            xlags = None
        # [n, m]
        elif len(window_length) == 2:
            ylags = window_length[0]
            xlags = window_length[1]
        else:
            raise ValueError(f"Invalid `windows_length`: {window_length}")

        # n : None
        if ylags is None:
            ylags = []
        # n : int
        elif isinstance(ylags, int):
            ylags = list(range(0, ylags))

        # m : None
        if xlags is None:
            xlags = []
        # m, b: int, bool
        elif isinstance(xlags, int):
            xlags = list(range(0, xlags))

        self._ylags = ylags
        self._xlags = xlags
        self._tlags = None

        self.window_length = max(lmax(self._ylags), lmax(self._xlags)) + 1
        return

    def _compose_tlags(self, prediction_length: int):
        assert isinstance(prediction_length, int)

        tlags = list(range(1, prediction_length + 1))

        self._ylags = None
        self._xlags = None
        self._tlags = tlags

        self.window_length = max(self._tlags)
        return

    def __len__(self):
        return self.window_length

    @property
    def xlags(self):
        return self._xlags

    @property
    def ylags(self):
        return self._ylags

    @property
    def tlags(self):
        return self._tlags
# end


def _is_fh_seq(fh):
    """ Check if fh is [1,2,...n]"""
    return len(fh) == fh[-1]


# def _to_tlags(fh):
#     """
#     Convert fh into tlags:
#
#         fh = [1,2,3] -> tlags = [0,1,2]
#
#     """
#     if fh is None:
#         tlags = [1]
#     elif fh.is_relative:
#         tlags = list(fh)
#     else:
#         raise ValueError("fh is not relative")
#     return tlags


# ---------------------------------------------------------------------------
# TabularRegressorForecaster
# ---------------------------------------------------------------------------

class StrategyBasedRegressorForecaster(_Reducer):
    strategy = "any"


    _tags = {
        # "y_inner_mtype": "np.ndarray",
        # "X_inner_mtype": "np.ndarray",
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "scitype:y": "both",
        "requires-fh-in-fit": False,
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
        self.window_length_ = WindowLength(window_length, False)
        self.prediction_length_ = WindowLength(prediction_length, True)

        self._lt = None
        self._pt = None
        pass

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def _transform(self, y, X=None) -> tuple[np.ndarray, np.ndarray]:
        xlags = self.window_length_.xlags
        ylags = self.window_length_.ylags
        tlags = self.prediction_length_.tlags

        lt = LinearTrainTransform(xlags=xlags, ylags=ylags, tlags=tlags)
        self._lt = lt
        self._pt = lt.predict_transform()

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
            yti = yt[:, i:i+1]

            estimator: ITabularEstimator = cast(ITabularEstimator, clone(self.estimator))
            estimator.fit(Xt, yti)

            self.estimators_.append(estimator)
    # end

    def _predict(self, fh, X):
        # fh MUST BE passed!
        # The check is done by 'sktime'
        self._check_X_y(X=X)
        self._check_fh(fh=fh)

        Xp = X
        yh = self._y
        Xh = self._X

        # n of timeslots to predict
        fh = cast(ForecastingHorizon, fh).to_relative(self.cutoff)
        nfh = len(fh)

        tlags = fh.to_numpy().tolist()

        pt = self._pt
        res = pt.fit(y=yh, X=Xh).transform(fh=fh, X=Xp)

        if self.strategy in ["direct", "dirrec"]:
            y_pred = self._predict_multiple(nfh, pt, tlags)
        elif self.strategy in ["recursive", "multioutput"]:
            y_pred = self._predict_single(nfh, pt)
        else:
            raise ValueError(f"Unsupported strategy {self.strategy}")

        return y_pred

    def _predict_single(self, nfh, pt):
        y_pred = pt._yp

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            yp: np.ndarray = self.estimator.predict(Xt)

            i = pt.update(i, yp)
        # end
        return y_pred

    def _predict_multiple(self, nfh, pt, tlags):
        y_pred = pt._yp

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

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------


class FlexibleDirectRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "direct"


class FlexibleRecursiveRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "recursive"


class FlexibleMultioutputRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "multioutput"


class FlexibleDirRecRegressionForecaster(StrategyBasedRegressorForecaster):
    strategy = "dirrec"
