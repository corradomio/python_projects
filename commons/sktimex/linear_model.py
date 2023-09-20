from datetime import datetime
from typing import Optional

import logging
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sktime.forecasting.base import BaseForecaster

from .model_transform import LinearTrainTransform, LinearPredictTransform
from .lag import resolve_lag
from .utils import *


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
                 lags: Union[int, list, tuple, dict] = (0, 1),
                 tlags=(0,),
                 estimator="sklearn.linear_model.LinearRegression",
                 **kwargs):
        """
        
        :param lags:
                int             same for input and target
                (ilag, tlag)    input lag, target lag
                {
                    'period_type': <period_type>,
                    'input': {
                        <period_type_1>: <count_1>,
                        <period_type_2>: <count_2>,
                        ...
                    },
                    'target: {
                        <period_type_1>: <count_1>,
                        <period_type_2>: <count_2>,
                        ...
                    },
                    'current': True
                }
                
        :param estimator:
        :param kwargs: 
        """
        super().__init__()
        self._class_name = estimator
        self._lags = lags
        self._kwargs = kwargs

        model_class = import_from(estimator)
        self._model = model_class(**kwargs)

        self._slots = resolve_lag(lags)
        self.Xh: Optional[np.ndarray] = None
        self.yh: Optional[np.ndarray] = None
        self._cutoff: Optional[datetime] = None

        p = estimator.rfind('.')
        self._log = logging.getLogger(f"LinearForecastRegressor.{estimator[p+1:]}")

        # self._scores: dict[str, float] = {}
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = {} | self._kwargs
        params['estimator'] = self._class_name
        params['lags'] = self._lags
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: Optional[ForecastingHorizon] = None):
        # DataFrame/Series -> np.ndarray
        # save only the s last slots (used in prediction)
        yf, Xf = self._validate_data_lfr(y, X)

        tt = LinearTrainTransform(slots=self._slots)
        Xt, yt = tt.fit_transform(X=Xf, y=yf)

        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning)
        #     self._model.fit(Xt, yt)
        self._model.fit(Xt, yt)
        self._save_history(yf, Xf)
        return self
    # end
    
    def _save_history(self, yf, Xf):
        s = len(self._slots)
        self.yh = yf[-s:] if yf is not None else None
        self.Xh = Xf[-s:] if Xf is not None else None
        pass

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    # predict(fh)   predict(fh, X)  predict(fh, X, y)
    #               predict(    X)  predict(    X, y)
    #

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None) -> Union[pd.DataFrame, pd.Series]:
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        # ensure fh relative
        fh = fh.to_relative(self.cutoff)

        # X, yh, Xh
        Xp, yh, Xh = self._validate_data_lfr(None, X, predict=True)
        """:type: np.ndarray, np.ndarray"""
        # n of slots to predict and populate y_pred
        n = int(fh[-1])

        pt = LinearPredictTransform(slots=self._slots)
        y_pred = pt.fit(X=Xh, y=yh).transform(X=Xp, fh=n)   # save X,y prediction

        for i in range(n):
            Xt = pt.step(i)
            yp: np.ndarray = self._model.predict(Xt)
            y_pred[i] = yp[0]

        # add the index
        y_pred: pd.Series = self._from_numpy(y_pred, fh)
        return y_pred
    # end

    def _from_numpy(self, ys: np.ndarray, fh: ForecastingHorizon) -> pd.Series:
        ys = ys.reshape(-1)
        y_index = fh.to_absolute(self.cutoff)
        yp = pd.Series(data=ys, index=y_index.to_pandas())
        return yp

    # def _make_fh_relative(self, fh: ForecastingHorizon):
    #     if fh is None:
    #         return None
    #     if not fh.is_relative:
    #         fh = fh.to_relative(self.cutoff)
    #     return fh

    # -----------------------------------------------------------------------
    # update
    # -----------------------------------------------------------------------

    def update(self, y, X=None, update_params=True):
        # Not necessary
        return self

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

        if predict and self.yh is None:
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

        if X is None and self.Xh is not None:
            raise ValueError(f"predict needs X")

        # yf, Xf
        if not predict:
            return y, X

        # Xf, yh, Xh
        if y is None:
            Xf = X
            yh = self.yh
            Xh = self.Xh
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

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state
    # end

    def __repr__(self, **kwargs):
        return f"LinearForecastRegressor[{self._model}]"

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
