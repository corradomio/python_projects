
__all__ = [
    "EarlyStopping",
    "compute_input_output_shapes",
    "NNX_RNN_FLAVOURS", "NNX_CNN_FLAVOURS", "NNX_LIN_FLAVOURS",
    "_BaseNNForecaster"
]

from typing import Union, Optional, Any

import numpy as np
import skorch
import torch

from torchx import nnlin as nnx, select_optimizer, select_criterion
from .base import TransformForecaster
from .base import yx_lags, t_lags
from ..utils import kwval, as_dict, qualified_name

# ---------------------------------------------------------------------------
# NN flavours
# ---------------------------------------------------------------------------

NNX_RNN_FLAVOURS = {
    None: nnx.LSTMLinear,            # default
    'lstm': nnx.LSTMLinear,
    'gru': nnx.GRULinear,
    'rnn': nnx.RNNLinear,
}


NNX_CNN_FLAVOURS = {
    None: nnx.Conv1dLinear,         # default
    'cnn': nnx.Conv1dLinear
}


NNX_LIN_FLAVOURS = {
    None: nnx.LinearEncoderDecoder, # default
    'lin': nnx.LinearEncoderDecoder
}


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

NNX_ENGINE_DEFAULTS = dict(
    criterion=qualified_name(torch.nn.MSELoss),
    optimizer=qualified_name(torch.optim.Adam),
    lr=0.01,
    batch_size=16,
    max_epochs=300,
    callbacks=None,
    patience=0,
)


class _BaseNNForecaster(TransformForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------
    #
    #   xlags, ylags, tlags
    #   model = {
    #       flavour: <flavour>
    #       activation: <name>
    #       activation_kwargs: {}
    #       **extra_params
    #   }
    #   nnengine = {
    #       criterion
    #       optimizer
    #       lr
    #       batch_size
    #       max_epochs
    #       callbacks
    #       patience,
    #       ** extra_params
    #   }

    def __init__(
        self, *,

        flavour,

        # -- time series
        lags: Union[int, list, tuple, dict],
        tlags: Union[int, list],

        # -- model
        # activation=None,
        # activation_kwargs=None,
        model: Optional[dict],

        # -- skorch
        engine: Optional[dict],

        # -- transform
        scaler: Union[None, str, dict],

        # criterion=None,
        # optimizer=None,
        # lr=0.01,
        # batch_size=16,
        # max_epochs=300,
        # callbacks=None,
        # patience=0,

        # -- extra parameters
        **kwargs
    ):
        super().__init__(
            scaler=scaler,
            # **kwargs
        )

        # Unmodified parameters [readonly]
        self.lags = lags
        self.tlags = tlags
        self.model = model
        self.engine = engine

        # self.flavour = flavour.lower() if isinstance(flavour, str) else flavour

        # self.criterion = criterion
        # self.optimizer = optimizer
        # self.callbacks = callbacks

        # Effective parameters
        self._model_params = model or {}
        self.flavour = flavour

        self._engine_params: dict[str, Any] = NNX_ENGINE_DEFAULTS | (engine or {})
        patience = kwval(self._engine_params, key="patience", defval=0)

        _yx_lags = yx_lags(lags)
        self._ylags = _yx_lags[0]
        self._xlags = _yx_lags[1]
        self._tlags = t_lags(tlags)

        if patience > 0:
            self._engine_params['callbacks'] = [EarlyStopping(patience=patience)]

        criterion = self._engine_params["criterion"]
        self._engine_params["criterion"] = select_criterion(criterion)

        optimizer = self._engine_params["optimizer"]
        self._engine_params["optimizer"] = select_optimizer(optimizer)

        #
        # skorch.NeuralNetRegressor configuration parameters
        #
        # self._nn_args = {
        #     'lags': lags,
        #     'tlags': tlags,
        #     'flavour': flavour,
        #     'activation': activation,
        #     'activation_kwargs': activation_kwargs,
        #     'patience': patience
        # }

        # WARN:
        #   these are the parameters passed to 'skorch'!
        #
        # self._skt_args = kwexclude(kwargs, ["method", "clip"]) | {
        #     "criterion": create_criterion(criterion),   # OVERRIDE
        #     "optimizer": create_optimizer(optimizer),   # OVERRIDE
        #     "lr": lr,
        #     "batch_size": batch_size,
        #     "max_epochs": max_epochs,
        #     "callbacks": callbacks                      # OVERRIDE
        # }

        #
        # Model instance
        #
        self._model = None
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    # def get_params(self, deep=True):
    #     # WARN:
    #     #   in 'skt_args' the parameters 'criterion' and 'optimizer'
    #     #   MUST BE override, because they are real classes
    #     # params = super().get_params(deep=deep) | self._skt_args | {
    #     #     'criterion': self.criterion,
    #     #     'optimizer': self.optimizer,
    #     #     'callbacks': self.callbacks,
    #     # } | self._nn_args # | self._kwargs
    #     params = super().get_params(deep=deep) | dict(
    #         lags=self.lags,
    #         tlags=self.tlags,
    #         model=self.model,
    #         engine=self.engine,
    #         scaler=self.scaler
    #     )
    #     return params
    # # end

    # -----------------------------------------------------------------------
    # update
    # -----------------------------------------------------------------------

    def _update(self, y, X=None, update_params=True):
        return super()._update(y=y, X=X, update_params=False)

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
        Xh = self._X
        yh = self._y

        return compute_input_output_shapes(Xh, yh, self._xlags, self._ylags, self._tlags)

    # def _from_numpy(self, ys, fhp):
    #     ys = ys.reshape(-1)
    #
    #     index = pd.period_range(self.cutoff[0] + 1, periods=len(ys))
    #     yp = pd.Series(ys, index=index)
    #     yp = yp.loc[fhp.to_pandas()]
    #     return yp.astype(float)

    # def _make_fh_relative_absolute(self, fh: ForecastingHorizon) -> tuple[ForecastingHorizon, ForecastingHorizon]:
    #     fhp = fh
    #     if fhp.is_relative:
    #         fh = fhp
    #         fhp = fh.to_absolute(self.cutoff)
    #     else:
    #         fh = fhp.to_relative(self.cutoff)
    #     return fh, fhp

    def _create_skorch_model(self, input_size, output_size):
        # Implemented in derived classes
        ...

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end
