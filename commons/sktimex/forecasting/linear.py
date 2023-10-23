import logging
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from .base import ExtendedBaseForecaster
from ..lags import resolve_lags, resolve_tlags
from ..transform.linear import LinearTrainTransform, LinearPredictTransform
from ..utils import PD_TYPES, to_matrix, import_from, qualified_name

__all__ = [
    "LinearForecastRegressor"
]


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

class LinearForecaster(ExtendedBaseForecaster):
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
                 estimator: Union[str, Any]="sklearn.linear_model.LinearRegression",
                 flatten=True,
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

        :param estimator: estimator to use. It can be
                - q fully qualified class name (str). The parameters to use must be passed with 'kwargs'
                - a Python class (type). The parameters to use must be passed with 'kwargs'
                - a class instance. The parameters 'kwargs' are retrieved from the instance
        :param kwargs: parameters to pass to the estimator constructor
        :param flatten: if to use a single model to predict the forecast horizon or a model for each
                timeslot
        """
        super().__init__()

        # Unmodified parameters [readonly]
        self.lags = lags
        self.tlags = tlags
        self.estimator = estimator
        self.flatten = flatten
        self.kwargs = kwargs

        # effective parameters
        self._kwargs = kwargs
        self._tlags = resolve_tlags(tlags)
        self._slots = resolve_lags(lags)

        self._models = {}       # one model for each 'tlag'
        self._model = None      # a single model for all 'tlags'

        if isinstance(estimator, str):
            self._class_name = estimator
            self._create_estimators(import_from(self._class_name))
        elif isinstance(estimator, type):
            self._class_name = qualified_name(estimator)
            self._create_estimators(estimator)
        else:
            self._class_name = qualified_name(type(estimator))
            self._kwargs = estimator.get_params()
            self._create_estimators(type(estimator))

        self.Xh: Optional[np.ndarray] = None
        self.yh: Optional[np.ndarray] = None

        name = self._class_name[self._class_name.rfind('.')+1:]
        self._log = logging.getLogger(f"LinearForecastRegressor.{name}")
        # self._log.info(f"Created {self}")
    # end

    def _create_estimators(self, estimator=None):
        if self.flatten:
            self._model = estimator(**self._kwargs)
        else:
            for t in self._tlags:
                self._models[t] = estimator(**self._kwargs)
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = {
            'lags': self.lags,
            'tlags': self.tlags,
            'flatten': self.flatten,
            'estimator': self.estimator
        }
        params = params | self._kwargs
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: Optional[ForecastingHorizon] = None):
        # DataFrame/Series -> np.ndarray
        # save only the s last slots (used in prediction)
        # yh, Xh = self._validate_data_lfr(y, X)
        # self._save_history(Xh, yh)

        Xh = to_matrix(X)
        yh = to_matrix(y)
        self._save_history(Xh, yh)

        if self.flatten:
            self._fit_flatten(Xh, yh)
        else:
            self._fit_tlags(Xh, yh)
        # end
        return self

    def _fit_flatten(self, Xh, yh):
        tt = LinearTrainTransform(slots=self._slots, tlags=self._tlags)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)
        self._model.fit(Xt, yt)

    def _fit_tlags(self, Xh, yh):
        tlags = self._tlags
        tt = LinearTrainTransform(slots=self._slots, tlags=tlags)
        Xt, ytt = tt.fit_transform(X=Xh, y=yh)
        st = len(tlags)

        for i in range(st):
            t = tlags[i]
            yt = ytt[:, i:i+1]
            self._models[t].fit(Xt, yt)

    def _save_history(self, Xh, yh):
        s = len(self._slots)
        self.Xh = Xh[-s:] if Xh is not None else None
        self.yh = yh[-s:] if yh is not None else None

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    # predict(fh)   predict(fh, X)  predict(fh, X, y)
    #               predict(    X)  predict(    X, y)
    #

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.

        fhp = fh
        if fhp.is_relative:
            fh = fhp
            fhp = fh.to_absolute(self.cutoff)
        else:
            fh = fhp.to_relative(self.cutoff)

        nfh = int(fh[-1])
        Xs = to_matrix(X)

        if self.flatten:
            y_pred = self._predict_flatten(Xs, nfh, fhp)
        else:
            y_pred = self._predict_tlags(Xs, nfh, fhp)

        return y_pred
    # end

    def _predict_flatten(self, Xs, nfh, fhp):
        pt = LinearPredictTransform(slots=self._slots, tlags=self._tlags)
        yp = pt.fit(X=self.Xh, y=self.yh).transform(X=Xs, fh=nfh)  # save X,y prediction

        for i in range(nfh):
            Xt = pt.step(i)

            y_pred: np.ndarray = self._model.predict(Xt)

            pt.update(i, y_pred)
        # end

        # add the index
        y_series: pd.Series = self._from_numpy(yp, fhp)
        return y_series

    def _predict_tlags(self, Xs, nfh, fhp):
        pt = LinearPredictTransform(slots=self._slots, tlags=self._tlags)
        yp = pt.fit(X=self.Xh, y=self.yh).transform(X=Xs, fh=nfh)  # save X,y prediction
        tlags = self._tlags

        i = 0
        while i < nfh:
            it = i
            for t in tlags:
                model = self._models[t]

                Xt = pt.step(i)

                y_pred: np.ndarray = model.predict(Xt)

                it = pt.update(i, y_pred, t)
            i = it
        # end

        # add the index
        y_series: pd.Series = self._from_numpy(yp, fhp)
        return y_series

    def _from_numpy(self, ys, fhp):
        ys = ys.reshape(-1)

        index = pd.period_range(self.cutoff[0] + 1, periods=len(ys))
        yp = pd.Series(ys, index=index)
        yp = yp.loc[fhp.to_pandas()]
        return yp.astype(float)

    # -----------------------------------------------------------------------
    # update
    # -----------------------------------------------------------------------

    def _update(self, y, X=None, update_params=True):
        return super()._update(y=y, X=X, update_params=False)

    # -----------------------------------------------------------------------
    # Implementation
    # -----------------------------------------------------------------------
    # # _validate_data is already defined in the superclass
    #
    # def _validate_data_lfr(self, y=None, X=None, predict: bool = False):
    #     # validate the data and converts it:
    #     #
    #     #   y in a np.array with rank 1
    #     #   X in a np.array with rank 2 OR None
    #     #  fh in a ForecastingHorizon
    #     #
    #     # Prediction cases
    #     #
    #     #   params      past        future      pred len
    #     #   fh          yh          -           |fh|
    #     #   X           Xh, yh      X           |X|
    #     #   fh, X       Xh, yh      X           |fh|
    #     #   fh, y       y           -           |fh|
    #     #   y, X        X[:y], y    X[y:]       |X|-|y|
    #     #   fh, y, X    X[:y], y    X[y:]       |fh|
    #     #
    #
    #     if predict and self.yh is None:
    #         raise ValueError(f'{self.__class__.__name__} not fitted yet')
    #
    #     # X to np.array
    #     if X is not None:
    #         if isinstance(X, pd.DataFrame):
    #             X = X.to_numpy()
    #         assert isinstance(X, np.ndarray)
    #         assert len(X.shape) == 2
    #
    #     # y to np.array
    #     if y is not None:
    #         if isinstance(y, pd.Series):
    #             y = y.to_numpy()
    #         elif isinstance(y, pd.DataFrame):
    #             assert y.shape[1] == 1
    #             self._cutoff = y.index[-1]
    #             y = y.to_numpy()
    #         assert isinstance(y, np.ndarray)
    #         if len(y.shape) == 2:
    #             y = y.reshape(-1)
    #         assert len(y.shape) == 1
    #
    #     if X is None and self.Xh is not None:
    #         raise ValueError(f"predict needs X")
    #
    #     # yf, Xf
    #     if not predict:
    #         return y, X
    #
    #     # Xf, yh, Xh
    #     if y is None:
    #         Xf = X
    #         yh = self.yh
    #         Xh = self.Xh
    #     else:
    #         n = len(y)
    #         yh = y
    #         Xh = X[:n]
    #         Xf = X[n:]
    #     return Xf, yh, Xh
    # # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state

    def __repr__(self, **kwargs):
        return f"LinearForecastRegressor[{self._class_name}]"

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


LinearForecastRegressor = LinearForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
