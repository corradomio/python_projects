from datetime import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd
import sktime.forecasting.base as skf
from pandas import PeriodIndex
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from stdlib import import_from

from .lag import resolve_lag, LagTrainTransform, LagPredictTransform


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FH_TYPES = Union[None, int, list[int], np.ndarray, ForecastingHorizon]


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
                 target: Optional[str] = None,
                 **kwargs):
        super().__init__()
        self._class_name = class_name
        self._lag = lag
        self._kwargs = kwargs

        model_class = import_from(class_name)
        self._model = model_class(**kwargs)

        self._target = target
        self._X_history: Optional[np.ndarray] = None
        self._y_history: Optional[np.ndarray] = None
        self._cutoff: Optional[datetime] = None
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = {} | self._kwargs
        params['class_name'] = self._class_name
        params['lag'] = self._lag
        params['target'] = self._target
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, y, X=None, fh=None):
        self._save_target(y)
        return super().fit(y=y, X=X, fh=fh)
    # end

    def _fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None, fh: Optional[ForecastingHorizon] = None):
        slots = resolve_lag(self._lag)
        s = len(slots)

        # DataFrame/Series -> np.ndarray
        # save only the s last slots (used in prediction)
        yf, Xf = self._validate_data_lfr(y, X)
        self._y_history = yf[-s:] if yf is not None else None
        self._X_history = Xf[-s:] if Xf is not None else None

        ltt = LagTrainTransform(slots=slots)
        Xt, yt = ltt.fit_transform(X=Xf, y=yf)

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning)
        #     self._model.fit(Xt, yt)
        self._model.fit(Xt, yt)

        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    # predict(fh)   predict(fh, X)  predict(fh, X, y)
    #               predict(    X)  predict(    X, y)
    #

    def predict(self,
                fh: FH_TYPES = None,
                X: Optional[pd.DataFrame] = None,
                y: Union[None, pd.DataFrame, pd.Series] = None):
        fh = self._resolve_fh(y, X, fh)
        if y is None:
            X = self._clip_on_cutoff(X)
            return super().predict(fh=fh, X=X)
        else:
            return self._predict(fh=fh, X=X, y=y)

    def _resolve_fh(self, y, X, fh: FH_TYPES) -> ForecastingHorizon:
        # (_, _, fh)        -> fh
        # (X, None, None)   -> |X|
        # (None, y, None)   -> error
        # (X, y, None)      -> |X| - |y|

        if fh is not None:
            cutoff = self.cutoff if y is None else y.index[-1]
            fh = fh if isinstance(fh, ForecastingHorizon) else ForecastingHorizon(fh)
            return fh.to_relative(cutoff)
        if y is None:
            n = len(X)
            return ForecastingHorizon(np.arange(1, n+1))
        else:
            n = len(X) - len(y)
            return ForecastingHorizon(np.arange(1, n+1))
    # end

    def _clip_on_cutoff(self, X):
        cutoff = self._cutoff[0] if isinstance(self._cutoff, PeriodIndex) else self._cutoff
        if X.index[0] <= cutoff:
            X = X.loc[X.index > cutoff]
        return X
    # end

    def _predict(self,
                fh: Optional[ForecastingHorizon],
                X: Optional[pd.DataFrame] = None,
                y: Union[None, pd.DataFrame, pd.Series] = None) -> pd.DataFrame:
        # fh is not None and it is relative!
        # normalize fh, y, X
        assert fh.is_relative
        slots = resolve_lag(self._lag)
        # X, yh, Xh
        Xp, yh, Xh = self._validate_data_lfr(y, X, predict=True)
        """:type: np.ndarray, np.ndarray"""
        # n of slots to predict and populate y_pred
        n = fh[-1]
        y_pred: np.ndarray = np.zeros(n)

        lpt = LagPredictTransform(slots=slots)
        lpt.fit(X=Xh, y=yh)             # save X,y history
        lpt.transform(X=Xp, y=y_pred)   # save X,y prediction

        for i in range(n):
            Xt = lpt.prepare(i)
            yp: np.ndarray = self._model.predict(Xt)
            y_pred[i] = yp[0]

        # add the index
        cutoff = self.cutoff if y is None else y.index[-1]
        y_pred = y_pred[fh-1]
        index = fh.to_absolute(cutoff).to_pandas()
        return pd.DataFrame(data=y_pred, columns=self._target, index=index)
    # end

    # -----------------------------------------------------------------------
    # score (not implemented yet)
    # -----------------------------------------------------------------------

    def score(self, y_true, X=None, fh=None) -> dict[str, float]:
        y_pred = self.predict(fh=fh, X=X)
        return {
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    # end

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------
    # _validate_data is already defined in the superclass

    def _save_target(self, y):
        if isinstance(y, pd.Series):
            self._target = [y.name]
        elif isinstance(y, pd.DataFrame):
            self._target = list(y.columns)
        else:
            self._target = [None]
    # end

    def _validate_data_lfr(self, y=None, X=None, predict: bool = False):
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
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            elif isinstance(y, pd.DataFrame):
                assert y.shape[1] == 1
                self._cutoff = y.index[-1]
                y = y.to_numpy()
            assert isinstance(y, np.ndarray)
            if len(y.shape) == 2:
                y = y.reshape(-1)
            assert len(y.shape) == 1

        if X is None and self._X_history is not None:
            raise ValueError(f"predict needs X")

        # yf, Xf
        if not predict:
            return y, X

        # Xf, yh, Xh
        if y is None:
            Xf = X
            yh = self._y_history
            Xh = self._X_history
        else:
            n = len(y)
            yh = y
            Xh = X[:n]
            Xf = X[n:]
        return Xf, yh, Xh
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def set_scores(self, scores):
        self._scores = scores

    def get_scores(self):
        return self._scores

    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state
    # end

    def __repr__(self):
        return f"LinearForecastRegressor[{self._model}]"

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------

# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
