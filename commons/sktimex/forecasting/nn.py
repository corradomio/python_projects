from typing import Union

import numpy as np
import pandas as pd
import skorch
import torch
from sktime.forecasting.base import ForecastingHorizon

from torchx import nnlin as nnx
from .base import TransformForecaster
from ..lags import resolve_lags, resolve_tlags, LagSlots
from ..utils import import_from, qualified_name, to_matrix, kwexclude

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
# compute_input_output_shapes
# ---------------------------------------------------------------------------

def compute_input_output_shapes(X: np.ndarray, y: np.ndarray,
                                xlags: list[int], ylags: list[int], tlags: list[int]) \
        -> tuple[tuple[int, int], tuple[int, int]]:
    sx = len(xlags) if X is not None else []
    sy = len(ylags)
    st = len(tlags)

    mx = X.shape[1] if X is not None and sx > 0 else 0
    my = y.shape[1]

    return (sy, mx + my), (st, my)
# end


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


NNX_LIN_FLAVOURS = {
    'lin': nnx.LinearEncoderDecoder
}


def NoLog(*args, **kwargs):
    pass


class EarlyStopping(skorch.callbacks.EarlyStopping):

    def __init__(self,
                 monitor='valid_loss',
                 patience=5,
                 threshold=1e-4,
                 threshold_mode='rel',
                 lower_is_better=True,
                 sink=NoLog,
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
# BaseNNForecaster
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

class BaseNNForecaster(TransformForecaster):
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
                 lags: Union[int, list, tuple, dict],
                 tlags: Union[int, list],

                 # --

                 scale=False,
                 flavour=None,
                 # activation=None,
                 # activation_params=None,

                 # --

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,

                 # --

                 patience=0,
                 **kwargs):

        super().__init__()

        #
        self._lags = lags
        self._slots: LagSlots = resolve_lags(lags)
        self._tlags = resolve_tlags(tlags)
        self._scale = scale
        self._flavour = flavour.lower() if isinstance(flavour, str) else flavour

        # optimizer = parse_class(optimizer, torch.optim.Adam)
        # criterion = parse_class(criterion, torch.nn.MSELoss)
        if patience > 0:
            callbacks = [EarlyStopping(patience=patience)]

        self._optimizer = parse_class(optimizer, torch.optim.Adam)
        self._criterion = parse_class(criterion, torch.nn.MSELoss)
        self._callbacks = callbacks

        #
        # skorch.NeuralNetRegressor configuration parameters
        #
        self._nn_args = dict(
            lags=lags,
            tlags=tlags,
            flavour=flavour,
            scale=scale,
            # activation=activation,
            # activation_params=activation_params,
            patience=patience
        )

        self._skt_args = kwexclude(kwargs, "method")
        self._skt_args["criterion"] = qualified_name(criterion)
        self._skt_args["optimizer"] = qualified_name(optimizer)
        self._skt_args["lr"] = lr
        self._skt_args["batch_size"] = batch_size
        self._skt_args["max_epochs"] = max_epochs
        self._skt_args["callbacks"] = callbacks

        self._kwargs = kwargs

        #
        #
        #
        self._model = None

        # index
        self._X = None
        self._y = None
    # end

    def get_params(self, deep=True):
        params = {} | self._skt_args | self._nn_args
        return params
    # end

    # -----------------------------------------------------------------------
    # update
    # -----------------------------------------------------------------------

    def _update(self, y, X=None, update_params=True):
        return super()._update(y=y, X=X, update_params=False)

    # def update(self, y, X=None, update_params=True):
    #     self._save_history(X, y)

    # -----------------------------------------------------------------------
    #
    # -----------------------------------------------------------------------

    # def _update_fit(self, y, X):
    #     ...

    # def fit_predict(self, y, X=None, fh=None):
    #     ...

    # def score(self, y, X=None, fh=None):
    #     ...

    # -----------------------------------------------------------------------

    # def predict_quantiles(self, fh=None, X=None, alpha=None):
    #     ...

    # def predict_interval(self, fh=None, X=None, coverage=0.90):
    #     ...

    # def predict_var(self, fh=None, X=None, cov=False):
    #     ...

    # def predict_proba(self, fh=None, X=None, marginal=True):
    #     ...

    # def predict_residuals(self, y=None, X=None):
    #     ...

    # -----------------------------------------------------------------------

    # def update(self, y, X=None, update_params=True):
    #     ...

    # def update_predict(self, y, cv=None, X=None, update_params=True, reset_forecaster=True):
    #     ...

    # def update_predict_single(self, y=None, fh=None, X=None, update_params=True):
    #     ...

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def _compute_input_output_shapes(self):
        Xh = to_matrix(self._X)
        yh = to_matrix(self._y)

        return compute_input_output_shapes(Xh, yh, self._slots.xlags, self._slots.ylags, self._tlags)

    def _from_numpy(self, ys, fhp):
        ys = ys.reshape(-1)

        index = pd.period_range(self.cutoff[0] + 1, periods=len(ys))
        yp = pd.Series(ys, index=index)
        yp = yp.loc[fhp.to_pandas()]
        return yp.astype(float)

    def _make_fh_relative_absolute(self, fh: ForecastingHorizon) -> tuple[ForecastingHorizon, ForecastingHorizon]:
        fhp = fh
        if fhp.is_relative:
            fh = fhp
            fhp = fh.to_absolute(self.cutoff)
        else:
            fh = fhp.to_relative(self.cutoff)
        return fh, fhp

    def _create_skorch_model(self, input_size, output_size):
        # Implemented in derived classes
        ...

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
