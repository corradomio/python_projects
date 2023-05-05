import warnings
from datetime import datetime
from typing import Union, Optional, Sized

import numpy as np
import pandas as pd
import sktime.forecasting.base as skf
from sklearn.exceptions import ConvergenceWarning
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from stdlib import import_from

from .lag import resolve_lag, LagTrainTransform, LagPredictTransform


# ---------------------------------------------------------------------------
# Prediction history
# ---------------------------------------------------------------------------
# this class is used to generate the predictions
# Note: Xh, yh, X, y can be collected from 2 different sources
#
#   1) Xh, yh are the X, y used during the training
#   2) Xh, yh are passed in input on 'predict'.
#       In this case, we must have |y| < |X|, y is used as yh
#       and X is split in 2 parts
#
#       Xh = X[:y]      such that |X[:y]| == |y|
#       Xp = X[y:]      used for the predictions
#
# Prediction cases
#
#   params      past        future      pred len
#   fh          yh          -           |fh|
#   X           Xh, yh      X           |X|
#   fh, X       Xh, yh      X           |fh|
#   fh, y       y           -           |fh|
#   y, X        X[:y], y    X[y:]       |X|-|y|
#   fh, y, X    X[:y], y    X[y:]       |fh|
#

# ---------------------------------------------------------------------------
# LinearForecastRegressor
# ---------------------------------------------------------------------------

#
# We suppose that the dataset is ALREADY normalized.
# the ONLY information is to know the name of the target column'
#

class LinearForecastRegressor(BaseForecaster):

    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "univariate",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": False,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": False,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
    }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 class_name: str,
                 lag: Union[int, list, tuple, dict],
                 **kwargs):
        super().__init__()
        self._class_name = class_name
        self._lag = lag
        self._kwargs = kwargs

        self._slots = resolve_lag(lag)
        model_class = import_from(class_name)
        self._model = model_class(**kwargs)
        self._X_history: Optional[np.ndarray] = None
        self._y_history: Optional[np.ndarray] = None
        self._fh: Optional[skf.ForecastingHorizon] = None
        self._cutoff: Optional[datetime] = None
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def fh(self):
        return self._fh

    def get_params(self, deep=True):
        params = {} | self._kwargs
        params['class_name'] = self._class_name
        params['lag'] = self._lag
        return params

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, fh: Optional[ForecastingHorizon] = None):
        y, X, fh = self._validate_data_lfr(y, X, fh)
        self._save_history(y, X, fh)

        ltt = LagTrainTransform(slots=self._slots)
        Xt, yt = ltt.fit_transform(X=X, y=y)

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning)
        #     self._model.fit(Xt, yt)
        self._model.fit(Xt, yt)

        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self,
                fh: Optional[ForecastingHorizon] = None,
                X: Optional[pd.DataFrame] = None,
                y: Optional[pd.Series] = None) -> pd.Series:
        # normalize fh, y, X
        yp, Xp, fh = self._validate_data_lfr(y, X, fh, predict=True)
        """:type: np.ndarray, np.ndarray, skf.ForecastingHorizon"""

        fh = fh.to_relative(self.cutoff)
        n = self._prediction_length(fh=fh, y=yp, X=Xp)
        y_pred = np.zeros(n)

        lpt = LagPredictTransform(slots=self._slots)
        lpt.fit(X=self._X_history, y=self._y_history)
        lpt.transform(X=Xp, y=y_pred)

        for i in range(n):
            Xt = lpt.prepare(i)
            yp: np.ndarray = self._model.predict(Xt)
            y_pred[i] = yp[0, 0]

        return self._compose_predictions(y_pred, y, X, fh)
    # end

    def _compose_predictions(self, y_pred, y, X, fh: ForecastingHorizon):
        y_pred = y_pred[fh.to_numpy()-1]
        fh = fh.to_absolute(self.cutoff)
        return pd.Series(y_pred, index=fh.to_pandas())
        # if X is None or not isinstance(X, (pd.Series, pd.DataFrame)):
        #     return pd.Series(y_pred, index=fh.to_pandas())
        #
        # if len(X) == len(y_pred):
        #     return pd.Series(y_pred, index=X.index)
        # else:
        #     index = X.index
        #     s = len(y) if y is not None else 0
        #     n = len(y_pred)
        #     return pd.Series(y_pred, index=index[s:s+n])
    # end

    # -----------------------------------------------------------------------
    # score (not implemented yet)
    # -----------------------------------------------------------------------

    def score(self, y, X=None, fh=None) -> dict[str, float]:
        raise NotImplemented()
    # end

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------
    # _validate_data is already defined in the superclass

    def _validate_data_lfr(self, y=None, X=None, fh=None, predict: bool = False):
        # validate the data and converts it:
        #
        #   y in a np.array with rank 1
        #   X in a np.array with rank 2 OR None
        #  fh in a ForecastingHorizon
        #
        # Prediction cases
        #
        #   params      past        future      pred len
        #   fh          yh          -           |fh|
        #   X           Xh, yh      X           |X|
        #   fh, X       Xh, yh      X           |fh|
        #   fh, y       y           -           |fh|
        #   y, X        X[:y], y    X[y:]       |X|-|y|
        #   fh, y, X    X[:y], y    X[y:]       |fh|
        #

        self._fh = fh

        if predict and self._y_history is None:
            raise ValueError(f'{self.__class__.__name__} not fitted yet')

        # X to np.array
        if X is not None:
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            assert isinstance(X, np.ndarray)
            assert len(X.shape) == 2

        # y to np.array
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                self._cutoff = y.index[-1]
                y = y.to_numpy()
            assert isinstance(y, np.ndarray)
            if len(y.shape) == 2:
                y = y.reshape(-1)
            assert len(y.shape) == 1

        # fh to ForecastingHorizon
        if fh is not None:
            if not isinstance(fh, skf.ForecastingHorizon):
                fh = skf.ForecastingHorizon(fh)

        if fh is not None or not predict:
            return y, X, fh

        if X is not None and y is not None:
            n = len(X) - len(y)
            fh = skf.ForecastingHorizon(np.arange(n))
        elif X is not None:
            n = len(X)
            fh = skf.ForecastingHorizon(np.arange(n))
        else:
            raise ValueError("Why 'fh' is not defined ?")

        if X is None and self._X_history is not None:
            raise ValueError(f"predict needs X")

        return y, X, fh
    # end

    def _save_history(self, y: np.ndarray, X: Optional[np.ndarray], fh: Optional[skf.ForecastingHorizon]):
        s = len(self._slots)

        self._fh = fh

        # history length == 0: nothing is necessary
        if s == 0:
            self._X_history = None
            self._y_history = None
            return

        # data used with
        #   predict(fh)
        #   predict(fh, X)
        if X is not None:
            self._X_history = X[-s:]
        if y is not None:
            self._y_history = y[-s:]
    # end

    def _prediction_length(self,
            fh: Union[skf.ForecastingHorizon, Sized],
            y: Optional[np.ndarray],
            X: Optional[np.ndarray]) -> int:

        # Prediction cases
        #
        #   params      past        future      pred len
        #   fh          yh          -           |fh|
        #   X           Xh, yh      X           |X|
        #   fh, X       Xh, yh      X           |fh|
        #   fh, y       y           -           |fh|
        #   y, X        X[:y], y    X[y:]       |X|-|y|
        #   fh, y, X    X[:y], y    X[y:]       |fh|
        #

        # compute the 'prediction length', number of future slots to generate
        if X is not None:
            return len(X)
        if X is not None and y is not None:
            return len(X) - len(y)
        if fh is not None:
            return fh.to_numpy()[-1]
        else:
            raise ValueError(f"Unable to compute the prediction length: fh or X are not specified")
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self):
        return f"LinearForecastRegressor[{self._model}]"

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------

# end
