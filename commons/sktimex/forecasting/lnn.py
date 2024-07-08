
__all__ = [
    "LNNLinearForecaster",
]

import logging
from typing import Sized, cast, Union, Optional

from sktime.forecasting.base import ForecastingHorizon

from skorchx import NeuralNetRegressor
from skorchx.callbacks import PrintLog
from .nn import *
from ..transform.nn import NNTrainTransform, NNPredictTransform
from ..utils import PD_TYPES
from ..utils import kwval, kwmerge, kwexclude

# ---------------------------------------------------------------------------
# LinearNNForecaster
# ---------------------------------------------------------------------------

NNX_LIN_DEFAULTS = dict(
    activation=None,
    hidden_size=None
)


class LNNLinearForecaster(_BaseNNForecaster):

    def __init__(
        self, *,

        lags: Union[int, list, tuple, dict],
        tlags: Union[int, list],

        flavour="lin",

        activation=None,
        hidden_size=None,

        engine: Optional[dict] = None,
        scaler: Optional[dict] = None,
    ):
        """
        Simple linear model based on a tensor having:

            (*, |lags|,|input_features|+|target|)   as input and
            (*, |tlags|, |target|)                  as output

        Because input features and targets are keeped together, it is not possible
        to have ylags <> tlags

        :param lags: input/target lags
        :param tlags: target prediction lags

        :param flavour: type of Linear ('lin')
        :param model.activation: activation function
        :param model.activation_kwargs: parameter for the activation function

        :param model.hidden_size: size of the hidden layer. If not specified,
            the model is composed by a single linear layer, otherwise by 2

        :param engine.optimizer: class of the optimizer to use (default: Adam)
        :param engine.optimizer_kwargs:
        :param engine.criterion: class of the loss to use (default: MSLoss)
        :param engine.criterion_kwargs:
        :param engine.batch_size: batch size (default 16)
        :param engine.max_epochs: EPOCHS (default 300)

        :param scaler.method: how to scale the values
        """
        super().__init__(
            locals()
        )

        model = self._model_kwargs
        isinstance(kwval(model, "hidden_size"), Optional[int])

        self._log = logging.getLogger(f"LinearNNForecaster.{self.flavour}")
    # end

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def _fit(self, y, X=None, fh=None):

        input_shape, output_shape = self._compute_input_output_shapes()

        self._model = self._create_skorch_model(input_shape, output_shape)

        yh, Xh = self.transform(y, X)

        tt = NNTrainTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=False)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)

        self._model.fit(y=yt, X=Xt)
        return self
    # end

    def _create_skorch_model(self, input_shape, output_shape):
        lin_constructor = NNX_LIN_FLAVOURS[self.flavour]

        #
        # create the linear model
        #
        lin = lin_constructor(
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
        model = NeuralNetRegressor(
            module=lin,
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
        _, Xs  = self.transform(None, X)

        # fh, fhp = self._make_fh_relative_absolute(fh)
        # nfh = int(fh[-1])
        nfh = len(cast(Sized, fh))

        pt = NNPredictTransform(xlags=self._xlags, ylags=self._ylags, tlags=self._tlags, flatten=True)
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

    def __repr__(self, n_char_max: int = 700):
        return f"LinearNNForecaster[{self.flavour}]"
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

