
__all__ = [
    "RegressorForecaster",
]

import logging
from typing import Union, Any

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from ..base import BaseForecaster
from ...transform.lags import yxu_lags, t_lags
from ...transform.lint import LinearTrainTransform
from ...utils import PD_TYPES, NoneType
from stdlib import create_from, name_of, class_of


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
# LinearRegressorForecaster
# ---------------------------------------------------------------------------

class RegressorForecaster(BaseForecaster):
    _tags = {
        # "ignores-exogeneous-X": False,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
    }

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        lags: Union[int, list, tuple] = 10,
        tlags: Union[int, list] = 1,
        estimator: Union[str, dict] = "sklearn.linear_model.LinearRegression",
        flatten=True,
        debug=False
    ):
        """
        Sktime compatible forecaster based on Scikit models.
        It offers the same interface to other sktime forecasters, instead than to use
        'make_reduction'.
        It extends the flexibility of 'make_reduction' because it is possible to
        specify past & future lags not only as simple integers but using specific
        list of integers to use as offset respect the timeslot to predict.

        The parameter 'lags' is a unique parameter for 'y' and 'X".
        If it is specified the lags for X, and the input features are not available,
        it will be ignored

        There are 2 lags specifications:

            lags:   lags for the past (xlags, ylags, ulags)
            tlags:  lags for the future (target lags)

        The lags values can be:

            int         : represent the sequence [0,1,2,...,n-1] for ylags, xlags lags
                          and       the sequence [  1,2,...,n  ] for tlags

            list/tuple  : specific from the FIRST day to predict

        :param lags: lags for target (y) and input features (X)

                    int                 same for target (ylags) and input (xlags)
                    (ylags,)            only for target (ylags). For input (xlags) will be []
                    ([], xlags)         only for input (xlags). For target (ylags) will be []
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
        super().__init__()

        assert isinstance(lags, (int, list, tuple)), f"Invalid 'lags' value: {lags}"
        assert isinstance(tlags, (int, list, tuple)), f"Invalid 'tlags' value: {tlags}"
        assert isinstance(estimator, (str, type, dict)), f"Invalid 'estimator' value: {estimator}"
        assert isinstance(flatten, bool), f"Invalid 'flatten' value: {flatten}"

        # Unmodified parameters [readonly]
        self.lags = lags
        self.tlags = tlags
        self.estimator = estimator
        self.flatten = flatten
        self.debug = debug

        # Effective parameters
        xyulags = yxu_lags(lags)
        tlags = t_lags(tlags)
        self._ylags = xyulags[0]            # past y lags
        self._xlags = xyulags[1]            # past X lags
        self._ulags = xyulags[2]            # future X lags (ulags)
        self._tlags = tlags            # future y lags (tlags)

        self._estimators = {}               # one model for each 'tlag'
        self._create_estimators()

        lt = LinearTrainTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, ulags=self._ulags)
        self._ft = lt
        self._pt = lt.predict_transform()

        estimator_class = class_of(self.estimator)
        name = name_of(estimator_class)
        self._log = logging.getLogger(f"sktimex.LinearForecaster.{name}")
    # end

    def _create_estimators(self):
        if self.flatten:
            self._estimators[0] = create_from(self.estimator)
        else:
            for t in self._tlags:
                self._estimators[t] = create_from(self.estimator)
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y, X, fh):
        if self.flatten:
            self._fit_flatten(y, X)
        else:
            self._fit_tlags(y, X)
        return self

    def _fit_flatten(self, yh, Xh):
        ft = self._ft
        Xt, yt = ft.fit_transform(y=yh, X=Xh)
        # self._dump_train(Xt,yt)
        self._call_estimator(0, Xt, yt)
    # end

    def _dump_train(self, Xt: pd.DataFrame, yt: pd.Series):
        if not self.debug:
            return
        try:
            df = Xt.copy()
            df["target"] = yt
            df.to_csv("../src/data_dumps/regressor_train.csv")
        except Exception as e:
            pass

    def _fit_tlags(self, yh, Xh):
        tlags = self._tlags
        ft = self._ft
        Xt, ytt = ft.fit_transform(y=yh, X=Xh)
        st = len(tlags)

        for i in range(st):
            t = tlags[i]
            yt = ytt[:, i:i+1]
            self._call_estimator(t, Xt, yt)
        pass
    # end

    def _call_estimator(self, t, Xt, yt):
        # if isinstance(yt, np.ndarray) and len(yt.shape) == 2:
        #     yt = yt.ravel()
        # elif isinstance(yt, pd.DataFrame):
        #     assert yt.shape[1] == 1, f"Invalid 'yt' shape: {yt.shape}"
        #     target = yt.columns[0]
        #     yt = yt[target]
        assert isinstance(Xt, (NoneType, pd.DataFrame))
        assert isinstance(yt, (pd.Series, pd.DataFrame))
        self._estimators[t].fit(Xt, yt)

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------
    # fit(y)        fit(y, X)
    # predict(fh)   predict(fh, X)
    #

    def predict(self, fh=None, X=None):
        self._check_estimators()
        return super().predict(fh, X)

    def _check_estimators(self):
        # strange problem:
        # in 'theory' the key used in '_estimators' dictionary must be and integer
        # BUT with write & read using pickle, the dictionary key BECAME a string
        # WHY???
        ekeys = list(self._estimators.keys())
        for k in ekeys:
            if not isinstance(k, int):
                self._log.warning("Estimators key IS NOT AN INTEGER. Converted")
                self._estimators[int(k)] = self._estimators[k]
                del self._estimators[k]
        # end
    # end

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        super()._predict(fh, X)
        fh = fh if fh.is_relative else fh.to_relative(self.cutoff)

        if self.flatten:
            X_pred, y_pred = self._predict_flatten(X, fh)
        else:
            X_pred, y_pred = self._predict_tlags(X, fh)

        assert isinstance(y_pred, (pd.Series, pd.DataFrame))

        return y_pred
    # end

    def _predict_flatten(self, X, fh):
        yh, Xh = self._y, self._X
        _, Xs = None, X

        pt = self._pt
        Xp, yp = pt.fit(y=yh, X=Xh).transform(fh=fh, X=Xs)  # save X, y prediction

        i = 0
        nfh = len(fh)
        while i < nfh:
            Xt = pt.step(i)

            y_pred: np.ndarray = self._estimators[0].predict(Xt)

            i = pt.update(i, y_pred)
        # end

        return Xp, yp

    def _predict_tlags(self, X, nfh):
        yh, Xh = self._y, self._X
        _, Xs = None, X

        pt = self._pt
        Xp, yp = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)  # save X, y prediction
        tlags = self._tlags

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            it = 0
            for t in tlags:
                model = self._estimators[t]

                y_pred: np.ndarray = model.predict(Xt)

                pt.update(i, y_pred, t)
                it += 1
                if (i + it) >= nfh: break
            # end

            i += it
        # end

        return Xp, yp

    # -----------------------------------------------------------------------
    # update
    # -----------------------------------------------------------------------
    
    def update(self, y, X=None, update_params=False):
        self._y = None
        self._X = None
        super().update(y=y, X=X, update_params=update_params)
        return self

    def _update(self, y, X=None, update_params=True):
        super()._update(y=y, X=X, update_params=update_params)
        # Note: estimators contain ONLY the model coefficients
        return self

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self, **kwargs):
        estimator_class = class_of(self.estimator)
        name = name_of(estimator_class)
        return f"RegressorForecaster[{name}]"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        # lags: Union[int, list, tuple] = 10,
        # tlags: Union[int, list] = 1,
        # estimator: Union[str, Any] = "sklearn.linear_model.LinearRegression",
        # flatten=True
        params_list = [
            {},
            {"lags": 6, "tlags": 1},
            {"lags": 6, "tlags": 2},
            {"lags": [6, 0], "tlags": 1},
        ]
        return params_list

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
