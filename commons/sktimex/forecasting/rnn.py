from sktime.forecasting.base import ForecastingHorizon
from torch import nn as nn

from .nn import *
from sktimex.transform.rnn import RNNTrainTransform, RNNPredictTransform
from sktimex.transform.rnn_slots import RNNSlotsTrainTransform, RNNSlotsPredictTransform
from sktimex.utils import FH_TYPES, PD_TYPES, to_matrix
from skorchx.callbacks import PrintLog

__all__ = [
    "SimpleRNNForecaster",
    "MultiLagsRNNForecaster",
]


# ---------------------------------------------------------------------------
# BaseRNNForecaster
# ---------------------------------------------------------------------------

class BaseRNNForecaster(BaseNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, *,
                 lags: Union[int, list, tuple, dict] = (0, 1),
                 tlags: Union[int, list, tuple] = (0,),
                 scale=True,

                 flavour='lstm',
                 activation=None,
                 activation_params=None,

                 # -- RNN

                 hidden_size=1,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.,

                 # -- opt/loss

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,
                 patience=0,
                 **kwargs):
        """

        :param lags: input/target lags
        :param flavour: type of RNN ('lstm', 'gru', 'rnn')
        :param optimizer: class of the optimizer to use (default: Adam)
        :param criterion: class of the loss to use (default: MSLoss)
        :param batch_size: batch size (default 16)
        :param max_epochs: EPOCHS (default 300)
        :param hidden_size: number of RNN hidden layers
        :param num_layers: number of RNN layers
        :param bidirectional: if to use a bidirectional
        :param dropout: if to apply a dropout
        :param kwargs: other parameters
        """
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

        #
        # torchx.nn.LSTM configuration parameters
        #
        self._rnn_args = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': bidirectional,
            'dropout': dropout,
            'activation': activation,
            'activation_params': activation_params
        }
    # end

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
        pass

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        pass

    # -----------------------------------------------------------------------
    # support
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params = params | self._rnn_args
        return params
    # end

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# SimpleRNNForecaster
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

class SimpleRNNForecaster(BaseRNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = (1,)):
        Xh = to_matrix(X)
        yh = self._apply_scale(to_matrix(y))
        self._save_history(Xh, yh)

        input_size, output_size = self._compute_input_output_sizes()

        self._model = self._create_skorch_model(input_size, output_size)

        tt = RNNTrainTransform(slots=self._slots)
        Xt, yt = tt.fit_transform(Xh, yh)

        self._model.fit(Xt, yt)
        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.

        fh, fhp = self._make_fh_relative_absolute(fh)

        # fhp = fh
        # if fhp.is_relative:
        #     fh = fhp
        #     fhp = fh.to_absolute(self.cutoff)
        # else:
        #     fh = fhp.to_relative(self.cutoff)

        nfh = int(fh[-1])
        Xs = to_matrix(X)
        pt = RNNPredictTransform(slots=self._slots, tlags=[0])
        ys = pt.fit(self.Xh, self.yh).transform(Xs, fh=nfh)

        for i in range(nfh):
            Xt = pt.step(i)

            y_pred = self._model.predict(Xt)

            pt.update(i, y_pred)
        # end

        ys = self._inverse_scale(ys)
        yp = self._from_numpy(ys, fhp)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def _create_skorch_model(self, input_size, output_size):
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
        rnn_constructor = NNX_RNN_FLAVOURS[self._flavour]
        rnn = rnn_constructor(
            steps=len(self._slots.target),
            input_size=input_size,
            output_size=output_size,
            **self._rnn_args
        )

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorch.NeuralNetRegressor(
            module=rnn,
            **self._skt_args
        )
        model.set_params(callbacks__print_log=PrintLog(delay=3))

        return model
    # end

    def __repr__(self):
        return f"SimpleRNNForecaster[{self._flavour}]"
# end


# ---------------------------------------------------------------------------
# MultiLagsRNNForecaster
# ---------------------------------------------------------------------------

class MultiLagsRNNForecaster(BaseRNNForecaster):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
        Xh = to_matrix(X)
        yh = self._apply_scale(to_matrix(y))
        self._save_history(Xh, yh)

        input_size, output_size = self._compute_input_output_sizes()

        self._model = self._create_skorch_model(input_size, output_size)

        tt = RNNSlotsTrainTransform(slots=self._slots)
        Xt, yt = tt.fit_transform(Xh, yh)

        self._model.fit(Xt, yt)
        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.

        fh, fhp = self._make_fh_relative_absolute(fh)

        # fhp = fh
        # if fhp.is_relative:
        #     fh = fhp
        #     fhp = fh.to_absolute(self.cutoff)
        # else:
        #     fh = fhp.to_relative(self.cutoff)

        nfh = int(fh[-1])
        Xs = to_matrix(X)
        pt = RNNSlotsPredictTransform(slots=self._slots, tlags=self._tlags)
        ys = pt.fit(self.Xh, self.yh).transform(Xs, fh=nfh)

        for i in range(nfh):
            Xt = pt.step(i)

            y_pred = self._model.predict(Xt)

            pt.update(i, y_pred)
        # end

        ys = self._inverse_scale(ys)
        yp = self._from_numpy(ys, fhp)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def _compute_input_output_sizes(self):
        Xh = self.Xh
        yh = self.yh
        mx = Xh.shape[1] if Xh is not None else 0
        my = yh.shape[1]
        return (mx, my), my

    def _create_skorch_model(self, input_size, output_size):
        mx, my = input_size

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
        rnn_constructor = NNX_RNN_FLAVOURS[self._flavour]

        #
        # input models
        #
        input_models = []
        inner_size = 0

        xlags_list = self._slots.input_lists
        for xlags in xlags_list:
            rnn = rnn_constructor(
                steps=len(xlags),
                input_size=mx,
                output_size=-1,         # disable nn.Linear layer
                **self._rnn_args
            )
            inner_size += rnn.output_size

            input_models.append(rnn)

        ylags_list = self._slots.target_lists
        for ylags in ylags_list:
            rnn = rnn_constructor(
                steps=len(ylags),
                input_size=my,
                output_size=-1,         # disable nn.Linear layer
                **self._rnn_args
            )
            inner_size += rnn.output_size

            input_models.append(rnn)

        #
        # output model
        #
        output_model = nn.Linear(in_features=inner_size, out_features=output_size)

        #
        # compose the list of input models with the output model
        #
        inner_model = nnx.MultiInputs(input_models, output_model)

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorch.NeuralNetRegressor(
            module=inner_model,
            warm_start=True,
            **self._skt_args
        )
        model.set_params(callbacks__print_log=PrintLog(delay=3))

        return model
    # end

    def __repr__(self):
        return f"MultiLagsRNNForecaster[{self._flavour}]"
# end

