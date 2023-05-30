from typing import Union, Optional
import numpy as np
import sktimex as sktx
import numpyx as npx
import torchx.nn as nnx
from sktimex.utils import PD_TYPES, FH_TYPES

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

class DeepForecastRegressor:

    def __init__(self,
                 lag: Union[int, list, tuple, dict],
                 target: Optional[str] = None,

                 current: Optional[bool] = None,
                 y_only: bool = False,
                 steps: int = 1,
                 optimizer: Optional[dict] = None,
                 loss: Optional[dict] = None,
                 batch_size: int = 16,
                 epochs: int = 300,

                 hidden_layers: int = 1,
                 num_layers: int = 1,
                 bidirectional: bool = False,
                 dropout: float = 0,

                 **kwargs
                 ):

        self._lag = lag
        self._target = target
        self._current = current
        self._y_only = y_only

        self._steps = steps

        self._hidden_size = hidden_layers
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._dropout = dropout

        self._optimizer = optimizer
        self._loss = loss
        self._batch_size = batch_size
        self._epochs = epochs
        self._kwargs = kwargs

        lags = sktx.resolve_lag(lag, current)
        self.xlags = lags.input
        self.ylags = lags.target

        self._module = None
        self.Xh = None
        self.yh = None

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = {} | self._kwargs
        params['lag'] = self._lag
        params['target'] = self._target
        params['y_only'] = self._y_only

        params['steps'] = self._steps

        params['optimizer'] = self._optimizer
        params['loss'] = self._loss

        params['batch_size'] = self._batch_size
        params['epochs'] = self._epochs

        params['hidden_size'] = self._hidden_size
        params['num_layers'] = self._num_layers
        params['dropout'] = self._dropout
        params['bidirectional'] = self._bidirectional
        return params
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _normalize_xy(self, X, y):
        if X is None:
            n = len(y)
            X = np.zeros((n, 0))

        if len(y.shape) == 1:
            y = y.reshape((-1, 1))
        if len(X.shape) == 1:
            X = X.reshape((-1, 1))

        self.Xh = X
        self.yh = y

        return X, y

    def _input_output_sizes(self):
        xlags = self.xlags
        ylags = self.ylags
        mx = self.Xh.shape[1]
        my = self.yh.shape[1]
        input_size = mx * len(xlags) + my * len(ylags)
        return input_size, my


    def fit(self, y: PD_TYPES, X: PD_TYPES=None, fh: FH_TYPES=None, val=None):
        X, y = self._normalize_xy(X, y)

        # scaled, if necessary
        Xs = X
        ys = y

        input_size, output_size = self._input_output_sizes()

        self._module = nnx.Module(
            model=[
                nnx.LSTM(
                    input_size=input_size,
                    hidden_size=self._hidden_size,
                    output_size=output_size,
                    num_layers=self._num_layers,
                    dropout=self._dropout,
                    bidirectional=self._bidirectional),
            ],
            optimizer=self._optimizer,
            loss=self._loss,
            epochs=self._epochs,
            batch_size=self._batch_size
        )

        lu = npx.UnfoldLoop(self._steps, xlags=self.xlags, ylags=self.ylags)
        Xt, yt = lu.fit_transform(Xs, ys)

        if val is not None:
            Xv, yv = val

            # scaled, if necessary
            Xvs = Xv
            yvs = yv

            Xvt, yvt = lu.transform(Xvs, yvs)
            val = (Xvt, yvt)
        # end

        self._module.fit(Xt, yt, val=val)
        pass

    def predict(self, fh, X=None):
        if X is None:
            pass
        pass
    # end
# end
