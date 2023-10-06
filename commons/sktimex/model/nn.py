from typing import Union

import pandas as pd
import skorch
import torch

from numpyx.transformers import MinMaxScaler
from torchx import nn as nnx
from .base import ExtendedBaseForecaster
from ..lag import resolve_lag, LagSlots
from ..utils import import_from

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
# Adadelta
# Adagrad
# Adam
# AdamW
# SparseAdam
# Adamx
# ASGD
# LBFGS
# NAdam
# RAdam
# RMSprop
# Pprop
# SGD
# .


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------
# It extends the default 'skorch.callbacks.EarlyStopping' in the following way:
#
#   if the loss increase, wait for patience time,
#   BUT if the loss decrease, it continues
#
# This differs from the original behavior:
#
#   if the loss decrease, update the best loss
#   if the loss increase or decrease BUT it is greater than the BEST loss
#   wait for patience time
#

NNX_RNN_FLAVOURS = {
    'lstm': nnx.LSTMLinear,
    'gru': nnx.GRULinear,
    'rnn': nnx.RNNLinear,
}

NNX_CNN_FLAVOURS = {
    'cnn': nnx.Conv1dLinear
}


class EarlyStopping(skorch.callbacks.EarlyStopping):

    def __init__(self,
                 monitor='valid_loss',
                 patience=5,
                 threshold=1e-4,
                 threshold_mode='rel',
                 lower_is_better=True,
                 sink=(lambda x: None),
                 load_best=False):
        super().__init__(monitor, patience, threshold, threshold_mode, lower_is_better, sink, load_best)
    # end

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        is_improved = self._is_score_improved(current_score)
        super().on_epoch_end(net, **kwargs)
        if not is_improved:
            self.dynamic_threshold_ = self._calc_new_threshold(current_score)
# end


def parse_class(aclass, default_class):
    if aclass is None:
        return default_class
    elif isinstance(aclass, str):
        return import_from(aclass)
    else:
        return aclass


# ---------------------------------------------------------------------------
# SimpleNNForecaster
# ---------------------------------------------------------------------------
#
# Note: the configuration
#
#       lags = [1, 1], steps = 12
#
# is equivalent to
#
#       lags = [12, 12]
#

class SimpleNNForecaster(ExtendedBaseForecaster):
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

    def __init__(self, *,
                 lags: Union[int, list, tuple, dict] = (0, 1),
                 scale: bool=False,

                 # --

                 flavour=None,
                 activation=None,
                 activation_params=None,

                 # --

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,
                 patience=0,
                 **kwargs):

        super().__init__()

        self._flavour = flavour.lower() if isinstance(flavour, str) else flavour
        self._lags = lags
        self._scale = scale

        optimizer = parse_class(optimizer, torch.optim.Adam)
        criterion = parse_class(criterion, torch.nn.MSELoss)
        if patience > 0:
            callbacks = [EarlyStopping(patience=patience)]

        self._optimizer = optimizer
        self._criterion = criterion
        self._callbacks = callbacks

        #
        # skorch.NeuralNetRegressor configuration parameters
        #
        self._nn_args = dict(
            lags=lags,
            flavour=flavour,
            scale=scale,

            activation=activation,
            activation_params=activation_params,
            patience=patience
        )

        self._skt_args = {} | kwargs
        self._skt_args["criterion"] = criterion
        self._skt_args["optimizer"] = optimizer
        self._skt_args["lr"] = lr
        self._skt_args["batch_size"] = batch_size
        self._skt_args["max_epochs"] = max_epochs
        self._skt_args["callbacks"] = callbacks

        #
        #
        #
        self._slots: LagSlots = resolve_lag(lags)
        self._model = None

        # index
        self.Xh = None
        self.yh = None

        # scalers
        self._y_scaler = None
    # end

    def get_params(self, deep=True):
        params = {} | self._skt_args | self._nn_args
        return params
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def _save_history(self, Xh, yh):
        s = len(self._slots)
        self.Xh = Xh[-s:] if Xh is not None else None
        self.yh = yh[-s:] if yh is not None else None

    def _compute_input_output_sizes(self):
        Xh = self.Xh
        yh = self.yh
        sx = len(self._slots.input)
        mx = Xh.shape[1] if Xh is not None and sx > 0 else 0
        my = yh.shape[1]
        input_size = mx + my
        return input_size, my

    def _from_numpy(self, ys, fhp):
        ys = ys.reshape(-1)

        index = pd.period_range(self.cutoff[0] + 1, periods=len(ys))
        yp = pd.Series(ys, index=index)
        yp = yp.loc[fhp.to_pandas()]
        return yp.astype(float)

    def _apply_scale(self, y):
        if not self._scale:
            return y

        if self._y_scaler is None:
            self._y_scaler = MinMaxScaler()
            self._y_scaler.fit(y)

        if y is not None:
            y = self._y_scaler.transform(y)

        return y

    def _inverse_scale(self, y):
        if self._y_scaler is not None:
            y = self._y_scaler.inverse_transform(y)
        return y

    def _create_skorch_model(self, input_size, output_size):
        pass

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state
    # end

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end
