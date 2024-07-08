
__all__ = [
    "CNNLinearForecaster",
]


import logging
from typing import cast, Sized, Union, Optional

import skorchx
from skorchx.callbacks import PrintLog
from sktime.forecasting.base import ForecastingHorizon
from .nn import *
from ..transform.nn import NNTrainTransform, NNPredictTransform
from ..utils import FH_TYPES, PD_TYPES, mul_
from ..utils import kwval, kwmerge, kwexclude


# ---------------------------------------------------------------------------
# BaseCNNForecaster
# ---------------------------------------------------------------------------
# lags, scale, flavour
# activation, activation_kwargs
# hidden_size, kernel_size, stride, padding, dilation, groups=1,
# criterion, optimizer, lr
# batch_size, max_epochs, callbacks, patience,
# kwargs

NNX_CNN_DEFAULTS = dict(
    activation=None,

    hidden_size=1,
    kernel_size=1,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
)


# ---------------------------------------------------------------------------
# CNNLinearForecaster
# ---------------------------------------------------------------------------

class CNNLinearForecaster(_BaseNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self, *,

        lags: Union[int, list, tuple, dict],
        tlags: Union[int, list],

        flavour="cnn",

        activation=None,
        hidden_size=1,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,

        engine: Optional[dict] = None,
        scaler: Optional[dict] = None,
    ):
        """
        Simple RNN layer followed by a linear layer.
        The tensors have structure

            (*, |lags|,|input_features|+|target|)   as input and
            (*, |tlags|, |target|)                  as output

        Because input features and targets are together, it is not possible
        to have ylags <> tlags

        :param lags: input/target lags
        :param tlags: target prediction lags

        :param flavour: type of Linear ('lin')
        :param model.activation: activation function
        :param model.activation_kwargs: parameter for the activation function

        :param model.hidden_size:
        :param model.kernel_size:
        :param model.stride:
        :param model.stride:
        :param model.padding:
        :param model.dilation:
        :param model.groups:

        :param engine.optimizer: class of the optimizer to use (default: Adam)
        :param engine.optimizer_kwargs:
        :param engine.criterion: class of the loss to use (default: MSLoss)
        :param engine.criterion_kwargs:
        :param engine.batch_size: batch size (default 16)
        :param engine.max_epochs: EPOCHS (default 300)

        :param scaler.method: how to scale the values
        """
        super().__init__(
            # flavour=flavour,
            # lags=lags,
            # tlags=tlags,
            # model=model,
            # engine=engine,
            # scaler=scaler,
            locals()
        )

        # model = kwmerge(NNX_CNN_DEFAULTS, model)
        model = self._model_kwargs
        assert isinstance(kwval(model, "hidden_size"), int)
        assert isinstance(kwval(model, "kernel_size"), int)
        assert isinstance(kwval(model, "stride"), int)
        assert isinstance(kwval(model, "padding"), int)
        assert isinstance(kwval(model, "dilation"), int)
        assert isinstance(kwval(model, "groups"), int)

        self._log = logging.getLogger(f"CNNForecaster.{self.flavour}")
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
        #   kernel_size,
        #   stride=1,
        #   padding=0,
        #   dilation=1,
        #   groups=1

        cnn_constructor = NNX_CNN_FLAVOURS[self.flavour]
        cnn = cnn_constructor(
            input_size=input_shape,
            output_size=output_shape,
            **self._model_kwargs
        )

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorchx.NeuralNetRegressor(
            module=cnn,
            callbacks__print_log=PrintLog(
                sink=logging.getLogger(str(self)).info,
                delay=3),
            **kwexclude(self._engine_kwargs, ["patience"])
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
        return f"CNNLinearForecaster[{self.flavour}]"
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
