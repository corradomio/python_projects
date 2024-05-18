import logging
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon

from .base import ExtendedBaseForecaster
from ..lags import resolve_lags, resolve_tlags
from ..transform.lin import LinearTrainTransform, LinearPredictTransform
from ..utils import PD_TYPES, import_from, qualified_name
from ..utils import to_matrix

__all__ = [
    "LinearForecaster",
    # "LinearForecastRegressor"
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
# LinearForecaster
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
                 lags: Union[int, list, tuple, dict],
                 tlags: Union[int, list],
                 estimator: Union[str, Any] = "sklearn.linear_model.LinearRegression",
                 flatten=False,
                 **kwargs):
        """
        Sktime compatible forecaster based on Scikit models.
        It offers the same interface to other sktime forecasters, instead than to use
        'make_reduction'.
        It extends the flexibility of 'make_reduction' because it is possibile to
        specify past & future lags not only as simple integers but using specific
        list of integers to use as offset respect the timeslot to predict.

        Note: there is AN INCOMPATIBILITY to resolve:

            in sktime the FIRST timeslot to predict has t=1
            here      the FIRST timeslot to predict has t=0

            THIS MUST BE RESOLVED!

        :param lags:
                int                 same for input and target
                (ilag, tlag)        input lag, target lag
                ([ilags], [tlags])  selected input/target lags
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
                - a fully qualified class name (str). The parameters to use must be passed with 'kwargs'
                - a Python class (type). The parameters to use must be passed with 'kwargs'
                - a class instance. The parameters 'kwargs' are retrieved from the instance
        :param kwargs: parameters to pass to the estimator constructor, if necessary, or retrieved from the
                estimator instance
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

        self._estimators = {}       # one model for each 'tlag'

        if isinstance(estimator, str):
            self.estimator = estimator
            self._create_estimators(import_from(self.estimator))
        elif isinstance(estimator, type):
            self.estimator = qualified_name(estimator)
            self._create_estimators(estimator)
        else:
            self.estimator = qualified_name(type(estimator))
            self._kwargs = estimator.get_params()
            self._create_estimators(type(estimator))

        self._X = None
        self._y = None

        name = self.estimator[self.estimator.rfind('.')+1:]
        self._log = logging.getLogger(f"LinearForecaster.{name}")
        # self._log.info(f"Created {self}")
    # end

    def _create_estimators(self, estimator=None):
        if self.flatten:
            self._estimators[0] = estimator(**self._kwargs)
        else:
            for t in self._tlags:
                self._estimators[t] = estimator(**self._kwargs)
    # end

    def transform(self, y, X):
        X = to_matrix(X)
        y = to_matrix(y)
        return y, X

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep) | {
            'lags': self.lags,
            'tlags': self.tlags,
            'flatten': self.flatten,
            'estimator': self.estimator
        } | self._kwargs
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: Optional[ForecastingHorizon] = None):
        self._X = X
        self._y = y

        yh, Xh = self.transform(y, X)

        if self.flatten:
            self._fit_flatten(Xh, yh)
        else:
            self._fit_tlags(Xh, yh)
        return self

    def _fit_flatten(self, Xh, yh):
        tt = LinearTrainTransform(slots=self._slots, tlags=self._tlags)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)
        self._estimators[0].fit(Xt, yt)

    def _fit_tlags(self, Xh, yh):
        tlags = self._tlags
        tt = LinearTrainTransform(slots=self._slots, tlags=tlags)
        Xt, ytt = tt.fit_transform(X=Xh, y=yh)
        st = len(tlags)

        for i in range(st):
            t = tlags[i]
            yt = ytt[:, i:i+1]
            self._estimators[t].fit(Xt, yt)

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

        if self.flatten:
            y_pred = self._predict_flatten(X, nfh, fhp)
        else:
            y_pred = self._predict_tlags(X, nfh, fhp)

        return y_pred
    # end

    def _predict_flatten(self, X, nfh, fhp):
        yh, Xh = self.transform(self._y, self._X)
        _, Xs  = self.transform(None, X)

        pt = LinearPredictTransform(slots=self._slots, tlags=self._tlags)
        yp = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)  # save X,y prediction

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            y_pred: np.ndarray = self._estimators[0].predict(Xt)

            i = pt.update(i, y_pred)
        # end

        # add the index
        yp = self.inverse_transform(yp)
        y_series: pd.Series = self._from_numpy(yp, fhp)
        return y_series

    def _predict_tlags(self, X, nfh, fhp):
        yh, Xh = self.transform(self._y, self._X)
        _, Xs = self.transform(None, X)

        pt = LinearPredictTransform(slots=self._slots, tlags=self._tlags)
        yp = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)  # save X,y prediction
        tlags = self._tlags

        i = 0
        while i < nfh:
            it = i
            for t in tlags:
                model = self._estimators[t]

                Xt = pt.step(i)

                y_pred: np.ndarray = model.predict(Xt)

                it = pt.update(i, y_pred, t)
            i = it
        # end

        # add the index
        yp = self.inverse_transform(yp)
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
        for key in self._estimators:
            self._update_estimator(self._estimators[key], y=y, X=X, update_params=False)
        return super()._update(y=y, X=X, update_params=False)

    def _update_estimator(self, estimator, y, X=None, update_params=True):
        try:
            estimator.update(y=y, X=X, update_params=update_params)
        except:
            pass
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state

    def __repr__(self, **kwargs):
        return f"LinearForecaster[{self.estimator}]"

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# Compatibility
LinearForecastRegressor = LinearForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
