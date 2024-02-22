import logging
from .nn import *
from ..utils import FH_TYPES, PD_TYPES
from ..transform.rnn import RNNTrainTransform, RNNPredictTransform
from skorchx.callbacks import PrintLog

__all__ = [
    "LinearNNForecaster",
]


# ---------------------------------------------------------------------------
# LinearNNForecaster
# ---------------------------------------------------------------------------

class LinearNNForecaster(BaseNNForecaster):

    def __init__(self, *,
                 lags: Union[int, list, tuple, dict] = (0, 1),
                 tlags: Union[int, list, tuple] = (0,),
                 scale=False,

                 flavour='lin',
                 activation=None,
                 activation_params=None,

                 # -- linear

                 hidden_size=None,

                 # -- opt/loss

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,
                 patience=0,
                 **kwargs):
        super().__init__(
            lags=lags,
            tlags=tlags,
            scale=scale,
            flavour=flavour,

            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            callbacks=callbacks,

            activation=activation,
            activation_params=activation_params,

            patience=patience,
            **kwargs
        )
        self._lnn_args = {
            'hidden_size': hidden_size,
            'activation': activation,
            'activation_params': activation_params
        }

        self.hidden_size = hidden_size

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = (1,)):

        input_shape, output_shape = self._compute_input_output_shapes()

        self._model = self._create_skorch_model(input_shape, output_shape)

        yh, Xh = self.transform(y, X)

        tt = RNNTrainTransform(slots=self._slots, tlags=self._tlags, flatten=False)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)

        self._model.fit(y=yt, X=Xt)
        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
        yh, Xh = self.transform(self._y, self._X)
        _, Xs  = self.transform(None, X)

        fh, fhp = self._make_fh_relative_absolute(fh)

        nfh = int(fh[-1])
        pt = RNNPredictTransform(slots=self._slots, tlags=self._tlags, flatten=True)
        ys = pt.fit(y=yh, X=Xh).transform(fh=nfh, X=Xs)

        i = 0
        while i < nfh:
            Xt = pt.step(i)

            y_pred = self._model.predict(Xt)

            i = pt.update(i, y_pred)
        # end

        ys = self.inverse_transform(ys)
        yp = self._from_numpy(ys, fhp)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params = params | self._lnn_args
        return params
    # end

    def _create_skorch_model(self, input_shape, output_shape):
        lin_constructor = NNX_LIN_FLAVOURS[self._flavour]

        lin = lin_constructor(
            input_shape=input_shape,
            output_shape=output_shape,
            **self._lnn_args
        )

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorch.NeuralNetRegressor(
            module=lin,
            **self._skt_args
        )
        model.set_params(callbacks__print_log=PrintLog(
            sink=logging.getLogger(str(self)).info,
            delay=3))

        return model

    def __repr__(self):
        return f"LinearNNForecaster[{self._flavour}]"
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------

