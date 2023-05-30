# 'sktime' protocols
#
#       cutoff
#       fh
#
#       model.fit(y, X=None, fh=None)
#       model.predict(fh=None, X=None)
#
#       fit_predict(y, X=None, fh=None):
#
#       update(y, X=None, update_params=True)
#       update_predict(y, cv=None, X=None, update_params=True, reset_forecaster=True)
#       update_predict_single(y=None, fh=None, X=None, update_params=True)
#
# In 'Time Series Forecasting in Python', the library predict MULTIPLE
# predictions in a single step
# In 'sktime' it is used a 'autoregressive approach: it is predictied just a single
# next timeslot, then this timeslot (and k-1 previous) is used to predict the next,
# etc
#
# To create a Deep Learning model we need to CREATE the MODEL and to COMPILE it
# using a LOSS function AND an OPTIMIZER.

from datetime import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd
from sktime.forecasting.base import BaseForecaster

import torchx
from .lag import resolve_lag, LagTrainTransform


# ---------------------------------------------------------------------------
# DeppForecastRegressor
# ---------------------------------------------------------------------------
# model: how many layers, layer's types, layer's sizes
# optimizer: which optimizer to use
# loss: which loss function to use
#
# note: some 'predefined' models can have a 'name'
#
# nnx.LSTM
#   input_size      this depends on lagx, |X[0]| and |y[0]|
#   hidden_size     2*input_size
#   output_size=1
#   num_layers=1
#   bias=True
#   batch_first=True
#   dropout=0
#   bidirectional=False
#   proj_size =0
#
#

class DeepForecastRegressor(BaseForecaster):
    def __init__(self,
                 lag: Union[int, list, tuple, dict],
                 target: Optional[str] = None,

                 model: Optional[dict] = None,
                 optimizer: Optional[dict] = None,
                 loss: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        self._lag = lag
        self._target = target
        self._model_config = model
        self._optimizer = optimizer
        self._loss = loss
        self._kwargs = kwargs

        # it is not possible to create the NN now, because
        # the structure of X and y are not known
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
        params['lag'] = self._lag
        params['target'] = self._target
        params['model'] = self._model_config
        params['optimizer'] = self._optimizer
        params['loss'] = self._loss
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, y, X=None, fh=None):
        self._save_target(y)
        return super().fit(y=y, X=X, fh=fh)
    # end

    def _fit(self, y, X=None, fh=None):
        slots = resolve_lag(self._lag)
        s = len(slots)

        # DataFrame/Series -> np.ndarray
        # save only the s last slots (used in prediction)
        yf, Xf = self._validate_data_lfr(y, X)
        self._y_history = yf[-s:] if yf is not None else None
        self._X_history = Xf[-s:] if Xf is not None else None

        ltt = LagTrainTransform(slots=slots)
        Xt, yt = ltt.fit_transform(X=Xf, y=yf)

        self._create_model(Xt, yt)
        self._fit_model(Xt, yt)

        return self
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

    def _create_model(self, Xt: np.ndarray, yt: np.ndarray):
        xsize = 0 if Xt is None else Xt.shape[1]
        ysize = 1 if len(yt.shape) == 1 else yt.shape[1]

        self._model = torchx.ConfigurableModule(self._model_config, input_size=xsize, output_size=ysize)
    # end

    def _fit_model(self, Xt: np.ndarray, yt: np.ndarray):
        pass


    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
