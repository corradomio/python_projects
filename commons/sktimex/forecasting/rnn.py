
__all__ = [
    "RNNLinearForecaster",
]


import logging
import skorchx

from typing import Sized, cast, Union, Optional
from sktime.forecasting.base import ForecastingHorizon

from skorchx.callbacks import PrintLog
from .nn import *
from ..transform.nn import NNTrainTransform, NNPredictTransform
from ..utils import PD_TYPES, mul_
from ..utils import kwmerge, kwval, kwexclude


# ---------------------------------------------------------------------------
# BaseRNNForecaster
# ---------------------------------------------------------------------------

NNX_RNN_DEFAULTS = dict(
    activation=None,

    hidden_size=1,
    num_layers=1,
    bidirectional=False,
    dropout=0.,
)


# class _BaseRNNForecaster(_BaseNNForecaster):
#     """
#     This is a simple linear model based on a tensor having:
#
#         (*, |lags|,|input_features|+|target|)   as input and
#         (*, |tlags|, |target|)                  as output
#
#     Because input features and targets are together, it is not possible
#     to have ylags <> tlags
#
#     Model parameters:
#
#         flavour
#         activation
#         activation_kwargs | activation__<param>
#
#         hidden_size=1,
#         num_layers=1,
#         bidirectional=False,
#         dropout=0.,
#
#     """
#
#     # -----------------------------------------------------------------------
#     # Constructor
#     # -----------------------------------------------------------------------
#
#     def __init__(
#         self, *,
#
#         lags: Union[int, list, tuple, dict],
#         tlags: Union[int, list],
#
#         flavour="rnn",
#         model: Optional[dict] = None,
#         engine: Optional[dict] = None,
#         scaler: Optional[dict] = None,
#
#         # -- time series
#         # lags: Union[int, list, tuple, dict],
#         # tlags: Union[int, list],
#
#         # -- model
#         # flavour='lstm',
#         # activation=None,
#         # activation_kwargs=None,
#
#         # -- model/RNN
#
#         # hidden_size=1,
#         # num_layers=1,
#         # bidirectional=False,
#         # dropout=0.,
#
#         # -- skorch
#         # criterion=None,
#         # optimizer=None,
#         # lr=0.01,
#         # batch_size=16,
#         # max_epochs=300,
#         # callbacks=None,
#         # patience=0,
#
#         # -- extra params
#
#         # **kwargs
#     ):
#         """
#
#         :param lags: input/target lags
#         :param tlags: target prediction lags
#
#         :param flavour: type of RNN ('lstm', 'gru', 'rnn')
#         :param model.activation: activation function
#         :param model.activation_kwargs: parameter for the activation function
#
#         :param model.hidden_size: number of RNN hidden layers
#         :param model.num_layers: number of RNN layers
#         :param model.bidirectional: if to use a bidirectional
#         :param model.dropout: if to apply a dropout
#
#         :param engine.optimizer: class of the optimizer to use (default: Adam)
#         :param engine.criterion: class of the loss to use (default: MSLoss)
#         :param engine.batch_size: batch size (default 16)
#         :param engine.max_epochs: EPOCHS (default 300)
#
#         :param scaler.method: how to scale the values
#
#         """
#         super().__init__(
#             flavour=flavour,
#
#             lags=lags,
#             tlags=tlags,
#
#             model=model,
#             engine=engine,
#             scaler=scaler,
#
#             # lags=lags,
#             # tlags=tlags,
#
#             # flavour=flavour,
#             # activation=activation,
#             # activation_kwargs=activation_kwargs,
#
#             # criterion=criterion,
#             # optimizer=optimizer,
#             # lr=lr,
#             # batch_size=batch_size,
#             # max_epochs=max_epochs,
#             # callbacks=callbacks,
#             # patience=patience,
#
#             # **kwargs
#         )
#
#         model = kwmerge(NNX_RNN_DEFAULTS, model)
#         assert isinstance(kwval(model, "hidden_size"), int)
#         assert isinstance(kwval(model, "num_layers"), int)
#         assert isinstance(kwval(model, "bidirectional"), bool)
#         assert isinstance(kwval(model, "dropout"), (int, float))
#
#         #
#         # torchx.nn.LSTM configuration parameters
#         #
#         # self._rnn_args = {
#         #     'hidden_size': hidden_size,
#         #     'num_layers': num_layers,
#         #     'bidirectional': bidirectional,
#         #     'dropout': dropout,
#         # }
#
#         self._model_params = model
#
#         self._log = logging.getLogger(f"RNNForecaster.{self.flavour}")
#     # end
#
#     # -----------------------------------------------------------------------
#     # Properties
#     # -----------------------------------------------------------------------
#
#     # def get_params(self, deep=True):
#     #     params = super().get_params(deep=deep) | self._rnn_args
#     #     return params
#
#     # -----------------------------------------------------------------------
#     # Operations
#     # -----------------------------------------------------------------------
#
#     def _fit(self, y, X=None, fh=None):
#         pass
#
#     def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
#         pass
#
#     # -----------------------------------------------------------------------
#     # Support
#     # -----------------------------------------------------------------------
#
#     # -----------------------------------------------------------------------
#     # end
#     # -----------------------------------------------------------------------
# # end


# ---------------------------------------------------------------------------
# RNNLinearForecaster
# ---------------------------------------------------------------------------
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
# Note: the neural netword can be created ONLY during 'fit', because the NN structure
# depends on the configuration AND X, y dimensions/ranks
#

class RNNLinearForecaster(_BaseNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self, *,

        lags: Union[int, list, tuple, dict],
        tlags: Union[int, list],

        flavour="rnn",
        model: Optional[dict] = None,
        engine: Optional[dict] = None,
        scaler: Optional[dict] = None,
    ):
        """
        Simple CNN layer followed by a linear layer.
        The tensors have structure

            (*, |lags|,|input_features|+|target|)   as input and
            (*, |tlags|, |target|)                  as output

        Because input features and targets are together, it is not possible
        to have ylags <> tlags

        :param lags: input/target lags
        :param tlags: target prediction lags

        :param flavour: type of RNN ('lstm', 'gru', 'rnn')
        :param model.activation: activation function
        :param model.activation_kwargs: parameter for the activation function

        :param model.hidden_size: number of RNN hidden layers
        :param model.num_layers: number of RNN layers
        :param model.bidirectional: if to use a bidirectional
        :param model.dropout: if to apply a dropout

        :param engine.optimizer: class of the optimizer to use (default: Adam)
        :param engine.criterion: class of the loss to use (default: MSLoss)
        :param engine.batch_size: batch size (default 16)
        :param engine.max_epochs: EPOCHS (default 300)

        :param scaler.method: how to scale the values

        """
        super().__init__(
            flavour=flavour,

            lags=lags,
            tlags=tlags,

            model=model,
            engine=engine,
            scaler=scaler,
        )

        model = kwmerge(NNX_RNN_DEFAULTS, model)
        assert isinstance(kwval(model, "hidden_size"), int)
        assert isinstance(kwval(model, "num_layers"), int)
        assert isinstance(kwval(model, "bidirectional"), bool)
        assert isinstance(kwval(model, "dropout"), (int, float))

        self._model_params = model

        self._log = logging.getLogger(f"RNNForecaster.{self.flavour}")
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):

        input_shape, output_shape = super()._compute_input_output_shapes()
        self._model = self._create_skorch_model(input_shape, output_shape)

        yh, Xh = self.transform(y, X)

        tt = NNTrainTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)

        self._model.fit(Xt, yt)
        return self
    # end

    def _create_skorch_model(self, input_shape, output_shape):
        # create the torch model
        #   input_size      this depends on xlags, |X[0]| and |y[0]|
        #   hidden_size     2*input_size
        #   output_size=1
        #   num_layers=1
        #   bias=True
        #   batch_first=True
        #   dropout=0
        #   bidirectional=False
        #   proj_size =0

        # input_shape  = input_size | (seqlen, input_size)
        # output_shape = seqlen*output_size | (seqlen, output_size)

        rnn_constructor = NNX_RNN_FLAVOURS[self.flavour]
        rnn = rnn_constructor(
            input_shape=input_shape,
            output_shape=output_shape,
            **self._model_params
        )

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorchx.NeuralNetRegressor(
            module=rnn,
            callbacks__print_log=PrintLog(
                sink=logging.getLogger(str(self)).info,
                delay=3),
            **kwexclude(self._engine_params, ["patience"])
        )
        # model.set_params(callbacks__print_log=PrintLog(
        #     sink=logging.getLogger(str(self)).info,
        #     delay=3))

        return model
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        yh, Xh = self.transform(self._y, self._X)
        _, Xs = self.transform(None, X)

        # fh, fhp = self._make_fh_relative_absolute(fh)
        # nfh = int(fh[-1])
        nfh = len(cast(Sized, fh))

        pt = NNPredictTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags)
        ys = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            y_pred = self._model.predict(Xt)

            i = pt.update(i, y_pred)
        # end

        yp = self.inverse_transform(ys)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self):
        return f"RNNLinearForecaster[{self.flavour}]"
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

