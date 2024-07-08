
__all__ = [
    "EarlyStopping",
    "compute_input_output_shapes",
    "NNX_RNN_FLAVOURS", "NNX_CNN_FLAVOURS", "NNX_LIN_FLAVOURS",
    "_BaseNNForecaster"
]

from typing import Union, Optional, Any

import skorch
import torch

from torchx import nnlin as nnx, select_optimizer, select_criterion
from .base import TransformForecaster
from ..transform import yx_lags, t_lags, compute_input_output_shapes
from ..utils import kwval, qualified_name

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

# def compute_input_output_shapes(
#     X: np.ndarray,
#     y: np.ndarray,
#     xlags: Union[list[int], RangeType],
#     ylags: Union[list[int], RangeType],
#     tlags: Union[list[int], RangeType]
# ) -> tuple[tuple[int, int], tuple[int, int]]:
#
#     sx = len(xlags) if X is not None else []
#     sy = len(ylags)
#     st = len(tlags)
#
#     mx = X.shape[1] if X is not None and sx > 0 else 0
#     my = y.shape[1]
#
#     return (sy, mx + my), (st, my)
# # end


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
    patience=10,
)


class _BaseNNForecaster(TransformForecaster):

    def __init__(self, locals):
        super().__init__()

        self._model_kwargs = {}
        self._engine_kwargs = {}

        for k in locals:
            if k in ['self', '__class__']:
                continue
            elif k == 'lags':
                self.lags = locals[k]
            elif k == 'tlags':
                self.tlags = locals[k]
            elif k == 'flavour':
                self.flavour = locals[k]
            elif k == 'engine':
                self._engine_kwargs = NNX_ENGINE_DEFAULTS | (locals[k] or {})
            elif k == 'scaler':
                self.scaler = locals[k]
            else:
                self._model_kwargs[k] = locals[k]
            setattr(self, k, locals[k])
        # end

        patience = kwval(self._engine_kwargs, key="patience", defval=0)

        yxlags = yx_lags(self.lags)
        self._ylags = yxlags[0]
        self._xlags = yxlags[1]
        self._tlags = t_lags(self.tlags)

        if patience > 0:
            self._engine_kwargs['callbacks'] = [EarlyStopping(patience=patience)]

        criterion = self._engine_kwargs["criterion"]
        self._engine_kwargs["criterion"] = select_criterion(criterion)

        optimizer = self._engine_kwargs["optimizer"]
        self._engine_kwargs["optimizer"] = select_optimizer(optimizer)

        self._model = None
    # end

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

    def _create_skorch_model(self, input_size, output_size):
        # Implemented in derived classes
        ...

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


class _BaseNNForecasterOld(TransformForecaster):

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
    #   nengine = {
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

        # -- transform
        scaler: Union[None, str, dict],

        # -- skorch
        engine: Optional[dict],

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
        self.lags = lags        # y/X lags
        self.tlags = tlags      # t)arget lags
        self.flavour = flavour  # model's flavour
        self.model = model      # model's configuration parameters
        self.engine = engine    # skorch/torch configuration parameters

        # Effective parameters
        self._model_params = model or {}

        self._engine_params: dict[str, Any] = NNX_ENGINE_DEFAULTS | (engine or {})

        patience = kwval(self._engine_params, key="patience", defval=0)

        yxlags = yx_lags(lags)
        self._ylags = yxlags[0]
        self._xlags = yxlags[1]
        self._tlags = t_lags(tlags)

        if patience > 0:
            self._engine_params['callbacks'] = [EarlyStopping(patience=patience)]

        criterion = self._engine_params["criterion"]
        self._engine_params["criterion"] = select_criterion(criterion)

        optimizer = self._engine_params["optimizer"]
        self._engine_params["optimizer"] = select_optimizer(optimizer)

        #
        # Model instance
        #
        self._model = None
    # end

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

    def _create_skorch_model(self, input_size, output_size):
        # Implemented in derived classes
        ...

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end
