import logging

from skorchx.callbacks import PrintLog
from .nn import *
from ..transform.rnn import RNNTrainTransform, RNNPredictTransform
from ..utils import FH_TYPES, PD_TYPES, mul_

__all__ = [
    "CNNLinearForecaster",
    # "MultiLagsCNNForecaster",
]


# ---------------------------------------------------------------------------
# BaseCNNForecaster
# ---------------------------------------------------------------------------
# lags, scale, flavour
# activation, activation_params
# hidden_size, kernel_size, stride, padding, dilation, groups=1,
# criterion, optimizer, lr
# batch_size, max_epochs, callbacks, patience,
# kwargs

class BaseCNNForecaster(BaseNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, *,
                 lags: Union[int, list, tuple, dict],
                 tlags: Union[int, list, tuple],
                 scale=True,

                 flavour='cnn',
                 activation=None,
                 activation_params=None,

                 # -- CNN

                 hidden_size=1,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,

                 # -- opt/loss

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,
                 patience=0):
        """

        :param lags: input/target lags
        :param flavour: type of CNN ('cnn')
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

            # activation=activation,
            # activation_params=activation_params,

            patience=patience
        )

        #
        # torchx.nn.Cond1d configuration parameters
        #
        self._cnn_args = {
            'hidden_size': hidden_size,
            'kernel_size': kernel_size,
            'stride': stride,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,

            'activation': activation,
            'activation_params': activation_params,
        }

        self._log = logging.getLogger(f"CNNForecaster.{flavour}")
    # end

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    # def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
    #     pass

    # def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
    #     pass

    # -----------------------------------------------------------------------
    # support
    # -----------------------------------------------------------------------

    def get_params(self, deep=True):
        params = super().get_params(deep=deep) | self._cnn_args
        return params
    # end

    def _compute_input_output_sizes(self):
        # (sx, mx+my), (st, my)
        input_shape, ouput_shape = super()._compute_input_output_shapes()
        return input_shape[1], mul_(ouput_shape)

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# CNNLinearForecaster
# ---------------------------------------------------------------------------

class CNNLinearForecaster(BaseCNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):

        # input_size, output_size = self._compute_input_output_sizes()
        # self._model = self._create_skorch_model(input_size, output_size)

        input_shape, output_shape = super()._compute_input_output_shapes()
        self._model = self._create_skorch_model(input_shape, output_shape)

        yh, Xh = self.transform(y, X)

        # tt = CNNTrainTransform(slots=self._slots, tlags=self._tlags, flatten=True)
        # tt = CNNTrainTransform(slots=self._slots, tlags=self._tlags)
        tt = RNNTrainTransform(slots=self._slots, tlags=self._tlags)
        Xt, yt = tt.fit_transform(X=Xh, y=yh)

        self._model.fit(Xt, yt)
        return self
    # end

    def _create_skorch_model(self, input_shape, output_shape):
        # create the torch model
        #   input_size      this depends on lagx, |X[0]| and |y[0]|
        #   hidden_size     2*input_size
        #   output_size=1
        #   kernel_size,
        #   stride=1,
        #   padding=0,
        #   dilation=1,
        #   groups=1

        cnn_constructor = NNX_CNN_FLAVOURS[self._flavour]
        cnn = cnn_constructor(
            input_size=input_shape,
            output_size=output_shape,
            **self._cnn_args
        )

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        model = skorch.NeuralNetRegressor(
            module=cnn,
            callbacks__print_log=PrintLog(
                sink=logging.getLogger(str(self)).info,
                delay=3),
            **self._skt_args
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

        fh, fhp = self._make_fh_relative_absolute(fh)

        nfh = int(fh[-1])
        # pt = CNNPredictTransform(slots=self._slots, tlags=self._tlags, flatten=True)
        # pt = CNNPredictTransform(slots=self._slots, tlags=self._tlags)
        pt = RNNPredictTransform(slots=self._slots, tlags=self._tlags)
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

    def __repr__(self):
        return f"CNNLinearForecaster[{self._flavour}]"
# end


# ---------------------------------------------------------------------------
# MultiLagsCNNForecaster
# ---------------------------------------------------------------------------

# class MultiLagsCNNForecaster(BaseCNNForecaster):
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     # -----------------------------------------------------------------------
#     # fit
#     # -----------------------------------------------------------------------
#
#     def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
#         # self._save_history(X, y)
#
#         yh, Xh = self.transform(y, X)
#
#         input_size, output_size = self._compute_input_output_sizes()
#
#         self._model = self._create_skorch_model(input_size, output_size)
#
#         tt = CNNSlotsTrainTransform(slots=self._slots, tlags=self._tlags)
#         Xt, yt = tt.fit_transform(Xh, yh)
#
#         self._model.fit(Xt, yt)
#         return self
#     # end
#
#     # -----------------------------------------------------------------------
#     # predict
#     # -----------------------------------------------------------------------
#
#     def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
#         # [BUG]
#         # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.
#
#         yh, Xh = self.transform(self._y, self._X)
#         _, Xs = self.transform(None, X)
#
#         fh, fhp = self._make_fh_relative_absolute(fh)
#
#         nfh = int(fh[-1])
#         pt = CNNSlotsPredictTransform(slots=self._slots, tlags=self._tlags)
#         ys = pt.fit(y=yh, X=Xh, ).transform(fh=nfh, X=Xs)
#
#         i = 0
#         while i < nfh:
#             Xt = pt.step(i)
#
#             y_pred = self._model.predict(Xt)
#
#             i = pt.update(i, y_pred)
#         # end
#
#         ys = self.inverse_transform(ys)
#         yp = self._from_numpy(ys, fhp)
#         return yp
#     # end
#
#     # -----------------------------------------------------------------------
#     # Support
#     # -----------------------------------------------------------------------
#
#     def _compute_input_output_sizes(self):
#         # (sx, mx+my), (st, my)
#         input_shape, ouput_shape = super()._compute_input_output_shapes()
#
#         my = ouput_shape[1]
#         mx = input_shape[1] - my
#
#         return (mx, my), my
#
#     def _create_skorch_model(self, input_size, output_size):
#         mx, my = input_size
#
#         # create the torch model
#         #   input_size      this depends on xlags, |X[0]| and |y[0]|
#         #   hidden_size     2*input_size
#         #   output_size=1
#         #   num_layers=1
#         #   bias=True
#         #   batch_first=True
#         #   dropout=0
#         #   bidirectional=False
#         #   proj_size =0
#         cnn_constructor = NNX_CNN_FLAVOURS[self._flavour]
#
#         #
#         # input models
#         #
#         input_models = []
#         inner_size = 0
#
#         xlags_lists = self._slots.xlags_lists
#         for xlags in xlags_lists:
#             cnn = cnn_constructor(
#                 steps=len(xlags),
#                 input_size=mx,
#                 output_size=-1,         # disable nn.Linear layer
#                 **self._cnn_args
#             )
#             inner_size += cnn.output_size
#
#             input_models.append(cnn)
#
#         ylags_lists = self._slots.ylags_lists
#         for ylags in ylags_lists:
#             cnn = cnn_constructor(
#                 steps=len(ylags),
#                 input_size=my,
#                 output_size=-1,         # disable nn.Linear layer
#                 **self._cnn_args
#             )
#             inner_size += cnn.output_size
#
#             input_models.append(cnn)
#
#         #
#         # output model
#         #
#         output_model = nn.Linear(in_features=inner_size, out_features=output_size)
#
#         #
#         # compose the list of input models with the output model
#         #
#         inner_model = nnx.MultiInputs(input_models, output_model)
#
#         # create the skorch model
#         #   module: torch module
#         #   criterion:  loss function
#         #   optimizer: optimizer
#         #   lr
#         #
#         model = skorch.NeuralNetRegressor(
#             module=inner_model,
#             warm_start=True,
#             **self._skt_args
#         )
#         model.set_params(callbacks__print_log=PrintLog(
#             sink=logging.getLogger(str(self)).info,
#             delay=3))
#
#         return model
#     # end
#
#     def __repr__(self):
#         return f"MultiLagsCNNForecaster[{self._flavour}]"
# # end
