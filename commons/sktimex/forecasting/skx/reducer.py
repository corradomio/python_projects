
__all__ = [
    'ReducerForecaster',
]

import logging
from typing import Literal

import pandas as pd
import numpy as np

from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from ..base import BaseForecaster
from ...utils import PD_TYPES, to_index, to_numpy, to_data
from stdlib import import_from, name_of, is_instance

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def validate_index(y_pred, y, cutoff, fh):
    assert isinstance(y_pred, (pd.DataFrame, pd.Series, np.ndarray))
    if isinstance(y, np.ndarray):
        return y_pred

    # if y_pred.index.equals(fh):
    #     return y_pred

    index = to_index(y, cutoff, fh)

    if isinstance(y_pred, pd.Series):
        y_pred.index = index
    elif isinstance(y_pred, pd.DataFrame):
        y_pred.set_index(index, inplace=True)
    else:
        raise ValueError(f"Unsupported type {type(y_pred)}")
    return y_pred


# ---------------------------------------------------------------------------
# ReducerForecaster
# ---------------------------------------------------------------------------
#     estimator,
#     strategy="recursive",
#     window_length=10,
#     prediction_length=1,
#     scitype="infer",
#     transformers=None,
#     pooling="local",
#     windows_identical=True

class ReducerForecaster(BaseForecaster):
    # _tags = {
    #     "capability:exogenous": True,
    #     "requires-fh-in-fit": False,
    #     "capability:missing_values": False,
    #     "capability:insample": False,
    # }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        estimator="sklearn.linear_model.LinearRegression",
        estimator_args=None,
        strategy: Literal["direct", "recursive", "multioutput", "dirrec"] = "recursive",
        window_length=10,
        prediction_length=1,
        scitype="infer",
        transformers=None,
        pooling="local",
        windows_identical=True
    ):
        assert isinstance(estimator, str)
        assert isinstance(window_length, (int, list, tuple))
        assert isinstance(prediction_length, int)
        assert isinstance(windows_identical, bool)
        assert is_instance(strategy, Literal["direct", "recursive", "multioutput", "dirrec"], f"Unsupported strategy {strategy}")

        super().__init__()

        self.estimator = estimator
        self.estimator_args = estimator_args
        self.strategy = strategy
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.scitype = scitype
        self.transformers = transformers
        self.pooling = pooling
        self.windows_identical = windows_identical

        self._is_sktime_make_reduction = make_reduction.__module__.startswith("sktime.")
        self._fh_pred = ForecastingHorizon(np.arange(1, self.prediction_length+1), is_relative=True)
        self._y_past = None
        self._X_past = None

        name = name_of(self.estimator)
        self._log = logging.getLogger(f"sktimex.ReducerForecaster.{name}")

        self._create_estimator(estimator_args or {})

        # self._log.info("__init__")
    # end

    def _create_estimator(self, kwargs):
        estimator_class = import_from(self.estimator)

        # create the scikit-learn regressor
        estimator = estimator_class(**kwargs)

        if self._is_sktime_make_reduction:
            # estimator,
            # strategy="recursive",
            # window_length=10,
            # scitype="infer",
            # transformers=None,
            # pooling="local",
            # windows_identical=True,
            self.estimator_ = make_reduction(
                estimator=estimator,
                strategy=self.strategy,
                transformers=self.transformers,
                window_length=self.window_length,
                windows_identical=self.windows_identical
            )
        else:
            # create the forecaster
            self.estimator_ = make_reduction(
                estimator=estimator,
                strategy=self.strategy,
                transformers=self.transformers,
                window_length=self.window_length,
                prediction_length=self.prediction_length,
                windows_identical=self.windows_identical
            )
        pass
    # end

    # -----------------------------------------------------------------------
    # fit/predict/update
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)       fit(y, X, fh)

    def _fit(self, y, X, fh=None):
        y = to_numpy(y, matrix=True)
        X = to_numpy(X)
        self.estimator_.fit(y=y, X=X, fh=self._fh_pred)
        return self

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        super()._predict(fh, X)
        X = to_numpy(X)
        fh = self._check_fh_prediction_length(fh)

        if self.strategy == "recursive":
            y_pred = self._predict_recursive(fh, X)
        else:
            y_pred = self._predict_window(fh, X)

        y_pred = to_data(y_pred, self._y, self.cutoff, fh)

        self._cleanup()

        return y_pred

    def _update(self, y, X=None, update_params=True):
        super()._update(y, X=X, update_params=update_params)
        # the tabular estimators don't have an 'update' method
        return self

    # -----------------------------------------------------------------------

    def _check_fh_prediction_length(self, fh: ForecastingHorizon):
        fh = fh.to_relative(self.cutoff) if fh and not fh.is_relative else fh if fh else self._fh_pred
        last = fh[-1]
        assert last % self.prediction_length == 0, f"Unsupported fh {fh}: is not multiple of ``prediction_length``"
        return fh

    def _predict_recursive(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        y_pred = self.estimator_.predict(X=X, fh=fh)
        return y_pred

    def _predict_window(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        last = fh[-1]

        ye = to_numpy(self.estimator_._y, matrix=True)
        Xe = to_numpy(self.estimator_._X)

        yh = to_numpy(self._y, matrix=True)
        Xh = to_numpy(self._X)
        yf = np.zeros((last,) + yh.shape[1:])
        Xf = to_numpy(X)

        at = 0
        while at < last:
            yp, Xp = self._compose_yx(at, yh, Xh, yf, Xf)

            self.estimator_.update(yp, Xp, update_params=False)
            y_pred = self.estimator_.predict(X=X, fh=self._fh_pred)

            yf[at:at+self.prediction_length] = y_pred[:]
            at += self.prediction_length
        # end

        self.estimator_.update(ye, Xe, update_params=False)

        selected = fh.to_numpy()-1
        if len(selected) > 1 or  selected[0] != [0]:
            pass
        yf = yf[selected]

        return yf
    # end

    def _compose_yx(self, at, yh, Xh, yf, Xf):
        if len(yh.shape) == 1:
            yh = yh.reshape(-1, 1)
        if len(yf.shape) == 1:
            yf = yf.reshape(-1, 1)

        if self._y_past is None:
            self._y_past = np.zeros((self.window_length,) + yh.shape[1:], dtype=yh.dtype)
            self._X_past = np.zeros((self.window_length,) + Xh.shape[1:], dtype=Xh.dtype) if self._X is not None else None
        # end

        y_past = self._y_past
        X_past = self._X_past

        past_length = self.window_length - at

        if past_length > 0:
            y_past[:past_length] = yh[-past_length:]
            y_past[past_length:] = yf[0:at]

        if X_past is not None and past_length > 0:
            X_past[:past_length] = Xh[-past_length:]
            X_past[past_length:] = Xf[0:at]

        return y_past, X_past

    def _cleanup(self):
        self._y_past = None
        self._X_past = None

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self):
        return f"ReducerForecaster[{name_of(self.estimator)}, {self.strategy}]"
        # if self.estimator_ is None:
        #     return f"ReducerForecaster[{name_of(self.estimator)}, {self.strategy}]"
        # else:
        #     return f"ReducerForecaster[{self.estimator_.__class__.__name__}[{name_of(self.estimator)}, {self.strategy}]]"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        params_list = [
            {},
            {"strategy": "direct"},
            {"strategy": "recursive"},
            {"strategy": "multioutput"},
            {"strategy": "multioutput"},
        ]
        return params_list

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
