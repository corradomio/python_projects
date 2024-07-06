
__all__ = [
    "LinearForecaster",
]


import logging
from typing import Optional, Union, Any, Sized, cast

import numpy as np
from .base import ForecastingHorizon, KwArgsForecaster
from ..transform import yx_lags, t_lags
from ..transform.lin import LinearTrainTransform, LinearPredictTransform
from ..utils import PD_TYPES, import_from, qualified_name, name_of


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
# LinearForecasterV2
# ---------------------------------------------------------------------------

class LinearForecaster(KwArgsForecaster):
    _tags = {
        "y_inner_mtype": "np.ndarray",
        "X_inner_mtype": "np.ndarray",
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
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
        It extends the flexibility of 'make_reduction' because it is possible to
        specify past & future lags not only as simple integers but using specific
        list of integers to use as offset respect the timeslot to predict.

        The parameter 'lags' is an unique parameter for 'y' and 'X".
        If it is specified the lags for X, and the input features are not available,
        it will be ignored

        | TODO: there is AN INCOMPATIBILITY to resolve:
        |
        |    in sktime the FIRST timeslot to predict has t=1
        |    here      the FIRST timeslot to predict has t=0
        |
        |    THIS MUST BE RESOLVED!

        There are 2 lags specifications:

            lags:   lags for the past (xlags, ylags)
            tlags:  lags for the future (target lags)

        The lags values can be:

            int         : represent the sequence [  1,2,...,n  ] for ylags, xlags lags
                          and       the sequence [0,1,2,...,n-1] for tlags

            list/tuple  : specific from the FIRST day to predict

        :param lags: lags for target (y) and input features (X)

                    int                 same for target (ylags) and input (xlags)
                    (ylags,)            only for target (ylags). For input (xlags) will be []
                    (ylag, xlag)        input lags, target lags

                ylags and xlags can ber represented with an integer, corresponding to the list
                [1,2,...*lags], or by a specific list of lags.
                The value None is converted into the empty list []

        :param tlags:  lags for the prdiction target (tlags). Same configuration as before

        :param estimator: estimator to use. It can be
                - a fully qualified class name (str). The parameters to use must be passed with 'kwargs'
                - a Python class (type). The parameters to use must be passed with 'kwargs'

        :param kwargs: parameters to pass to the estimator constructor

        :param flatten: if to use a single model to predict the forecast horizon or a model for each
                timeslot
        """
        super().__init__(**kwargs)

        # Unmodified parameters [readonly]
        self.lags = lags
        self.tlags = tlags
        self.estimator = qualified_name(estimator)
        self.flatten = flatten

        # Effective parameters
        _xylags = yx_lags(lags)
        self._ylags = _xylags[0]            # past y lags
        self._xlags = _xylags[1]            # past x lags
        self._tlags = t_lags(tlags)         # future lags (tlags)

        self._estimators = {}               # one model for each 'tlag'
        self._create_estimators(kwargs)

        name = name_of(self.estimator)
        self._log = logging.getLogger(f"LinearForecaster.{name}")
    # end

    def _create_estimators(self, kwargs):
        estimator = import_from(self.estimator)

        if self.flatten:
            self._estimators[0] = estimator(**kwargs)
        else:
            for t in self._tlags:
                self._estimators[t] = estimator(**kwargs)
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep) | {
            'lags': self.lags,
            'tlags': self.tlags,
            'estimator': self.estimator,
            'flatten': self.flatten,
        }   # | self._kwargs
        return params

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):
        assert isinstance(y, np.ndarray)

        if self.flatten:
            self._fit_flatten(y, X)
        else:
            self._fit_tlags(y, X)
        return self

    def _fit_flatten(self, yh, Xh):
        tt = LinearTrainTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=True)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)
        self._estimators[0].fit(Xt, yt)

    def _fit_tlags(self, yh, Xh):
        tlags = self._tlags
        tt = LinearTrainTransform(xlags=self._xlags, ylags=self._ylags, tlags=tlags, flatten=True)
        Xt, ytt = tt.fit_transform(X=Xh, y=yh)
        st = len(tlags)

        for i in range(st):
            t = tlags[i]
            yt = ytt[:, i:i+1]
            self._estimators[t].fit(Xt, yt)
        pass
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    # predict(fh)   predict(fh, X)  predict(fh, X, y)
    #               predict(    X)  predict(    X, y)
    #

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        nfh = len(cast(Sized, fh))

        if self.flatten:
            y_pred = self._predict_flatten(X, nfh)
        else:
            y_pred = self._predict_tlags(X, nfh)

        return y_pred
    # end

    def _predict_flatten(self, X, nfh):
        yh, Xh = self._y, self._X
        _, Xs = None, X

        pt = LinearPredictTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=True)
        yp = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)  # save X,y prediction

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            y_pred: np.ndarray = self._estimators[0].predict(Xt)

            i = pt.update(i, y_pred)
        # end

        return yp

    def _predict_tlags(self, X, nfh):
        yh, Xh = self._y, self._X
        _, Xs = None, X

        pt = LinearPredictTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=True)
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

                if it >= nfh: break
            i = it
        # end

        return yp

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

    # def get_state(self) -> bytes:
    #     import pickle
    #     state: bytes = pickle.dumps(self)
    #     return state

    def __repr__(self, **kwargs):
        return f"LinearForecaster[{name_of(self.estimator)}]"

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
