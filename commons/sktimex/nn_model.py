from typing import Union, Optional

import numpy as np
import pandas as pd
import skorch
import torch.optim
from sktime.forecasting.base import ForecastingHorizon, BaseForecaster

from torch import nn as nn
from torchx import nn as nnx

from .lag import resolve_lag, LagSlots
from .model_transform import RNNTrainTransform, RNNPredictTransform
from .model_transform import RNNSlotsTrainTransform, RNNSlotsPredictTransform
from .model_transform import CNNTrainTransform, CNNPredictTransform
from .utils import import_from
from .utils import PD_TYPES, FH_TYPES


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
    'lstm': nnx.LSTM,
    'gru': nnx.GRU,
    'rnn': nnx.RNN,
}

NNX_CNN_FLAVOURS = {
    'cnn': nnx.Conv1d
}


# def _nolog(*args, **kwargs):
#     pass


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

class SimpleNNForecaster(BaseForecaster):
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

                 # --

                 flavour=None,
                 activation='relu',
                 activation_params=None,

                 # --

                 criterion=None,
                 optimizer=None,
                 lr=0.01,

                 batch_size=16,
                 max_epochs=300,
                 callbacks=None,
                 patience=20,
                 **kwargs):

        super().__init__()

        self._flavour = flavour.lower() if isinstance(flavour, str) else flavour

        self._optimizer = optimizer
        self._criterion = criterion
        self._callbacks = callbacks

        self._lags = lags

        #
        # skorch.NeuralNetRegressor configuration parameters
        #

        self._skt_args = {} | kwargs
        self._skt_args["criterion"] = criterion  # replace!
        self._skt_args["optimizer"] = optimizer  # replace!
        self._skt_args["lr"] = lr
        self._skt_args["batch_size"] = batch_size
        self._skt_args["max_epochs"] = max_epochs
        self._skt_args["callbacks"] = callbacks  # replace!

        #
        #
        #

        self._slots: LagSlots = resolve_lag(lags)
        self._model = None

        # index
        self.Xh = None
        self.yh = None
    # end

    # -----------------------------------------------------------------------
    # update
    # -----------------------------------------------------------------------

    def update(self, y, X=None, update_params=True):
        # Not necessary
        #
        # return super().update(y=y, X=X, update_params=update_params)
        return self

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def _save_history(self):
        s = len(self._slots)
        Xh = self.Xh
        yh = self.yh
        self.Xh = Xh[-s:] if Xh is not None else None
        self.yh = yh[-s:] if yh is not None else None
    # end

    def _to_dataframe(self, X: PD_TYPES, y: PD_TYPES = None, fh: FH_TYPES = None):

        if X is None:
            if y is not None:
                X = pd.DataFrame({}, index=y.index)
            elif fh.is_relative:
                cutoff = self.cutoff
                fh = fh.to_absolute(cutoff)
                X = pd.DataFrame({}, index=fh.to_pandas())
            else:
                X = pd.DataFrame({}, index=fh.to_pandas())
        # end

        if isinstance(X, pd.Series):
            X = pd.DataFrame({"X": X}, index=X.index)
        if isinstance(y, pd.Series):
            y = pd.DataFrame({"y": y}, index=y.index)
        return X, y

    def _to_numpy(self, X, y=None):
        Xs = X.to_numpy().astype(np.float32)
        if y is not None:
            ys = y.to_numpy().astype(np.float32)
        else:
            ys = None

        if len(Xs.shape) == 1:
            Xs = Xs.reshape((-1, 1))
        if ys is not None and len(ys.shape) == 1:
            ys = ys.reshape((-1, 1))

        # ys is not none ONLY in train
        if ys is not None:
            self.Xh = Xs
            self.yh = ys

        return Xs, ys

    def _compute_input_output_sizes(self):
        sx = len(self._slots.input)
        mx = self.Xh.shape[1] if self.Xh is not None and sx > 0 else 0
        my = self.yh.shape[1]
        input_size = mx + my
        return input_size, my

    def _from_numpy(self, ys, fhp):
        ys = ys.reshape(-1)

        index = pd.period_range(self.cutoff[0] + 1, periods=len(ys))
        yp = pd.Series(ys, index=index)
        yp = yp.loc[fhp.to_pandas()]
        return yp.astype(float)
    # end

    def get_state(self) -> bytes:
        import pickle
        state: bytes = pickle.dumps(self)
        return state
    # end
# end


# ---------------------------------------------------------------------------
# SimpleCNNForecaster
# ---------------------------------------------------------------------------

class SimpleCNNForecaster(SimpleNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, *,
                 lags: Union[int, list, tuple, dict] = (0, 1),
                 flavour='cnn',
                 relu=None,
                 activation='relu',
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
                 patience=20,
                 **kwargs):
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
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            callbacks=callbacks,

            flavour=flavour,
            relu=relu,
            activation=activation,
            activation_params=activation_params
        )
        if relu:
            activation = 'relu'

        # self._optimizer = optimizer
        # self._criterion = criterion
        # self._callbacks = callbacks

        # some defaults
        if optimizer is None:
            optimizer = torch.optim.Adam
        if criterion is None:
            criterion = torch.nn.MSELoss
        if patience > 0:
            callbacks = [EarlyStopping(patience=patience)]

        # some classes specified as string
        if isinstance(optimizer, str):
            optimizer = import_from(optimizer)
        if isinstance(criterion, str):
            criterion = import_from(criterion)

        self._lags = lags

        #
        # torchx.nn.LSTM configuration parameters
        #
        # self._flavour = flavour.lower() if isinstance(flavour, str) else flavour
        self._hidden_size = hidden_size

        self._cnn_args = {}
        self._cnn_args['hidden_size'] = hidden_size
        self._cnn_args['kernel_size'] = kernel_size
        self._cnn_args['stride'] = stride
        self._cnn_args['padding'] = padding
        self._cnn_args['dilation'] = dilation
        self._cnn_args['groups'] = groups
        self._cnn_args['activation'] = activation
        self._cnn_args['activation_params'] = activation_params

        #
        # skorch.NeuralNetRegressor configuration parameters
        #
        self._skt_args = {} | kwargs
        self._skt_args["criterion"] = criterion     # replace!
        self._skt_args["optimizer"] = optimizer     # replace!
        self._skt_args["lr"] = lr
        self._skt_args["batch_size"] = batch_size
        self._skt_args["max_epochs"] = max_epochs
        self._skt_args["callbacks"] = callbacks     # replace!

        #
        #
        #

        # self._model = None

        # index
        # self.Xh = None
        # self.yh = None

    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def lags(self):
        return self._lags

    @property
    def flavour(self):
        return self._flavour

    @property
    def hidden_size(self):
        # return self._cnn_args['hidden_size']
        return self._hidden_size

    @property
    def activation(self):
        return self._cnn_args['activation']

    @property
    def activation_params(self):
        return self._cnn_args['activation_params']

    @property
    def kernel_size(self):
        return self._cnn_args['kernel_size']

    @property
    def stride(self):
        return self._cnn_args['stride']

    @property
    def padding(self):
        return self._cnn_args['padding']

    @property
    def dilation(self):
        return self._cnn_args['dilation']

    @property
    def groups(self):
        return self._cnn_args['groups']

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr(self):
        return self._skt_args["lr"]

    @property
    def batch_size(self):
        return self._skt_args["batch_size"]

    @property
    def max_epochs(self):
        return self._skt_args["max_epochs"]

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def patience(self):
        return self._skt_args["patience"]

    def get_params(self, deep=True, **kwargs):
        params = {} | self._skt_args | self._cnn_args
        params['lags'] = self._lags
        params['flavour'] = self._flavour

        params['hidden_size'] = self._hidden_size

        # convert 'criterion', 'optimizer', 'callbacks' in the original configuration
        params['criterion'] = self._criterion
        params['optimizer'] = self._optimizer
        params["callbacks"] = self._callbacks

        return params

    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
        # normalize X, y as 'pandas' objects
        X, y = self._to_dataframe(X, y)
        # normalize X, y as numpy objects
        Xs, ys = self._to_numpy(X, y)
        # evaluate the input_size/ouput_size
        input_size, output_size = self._compute_input_output_sizes()

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
            steps=len(self._slots.target),
            input_size=input_size,
            output_size=output_size,
            **self._cnn_args
        )

        # create the skorch model
        #   module: torch module
        #   criterion:  loss function
        #   optimizer: optimizer
        #   lr
        #
        self._model = skorch.NeuralNetRegressor(
            module=cnn,
            callbacks__print_log=None,
            **self._skt_args
        )

        #
        # prepare the data to pass the the Convolutional NN
        #
        tt = CNNTrainTransform(slots=self._slots)
        Xt, yt = tt.fit_transform(X=Xs, y=ys)

        self._model.fit(Xt, yt)
        self._save_history()
        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        fhp = fh
        if not fhp.is_relative:
            fh = fhp.to_relative(self.cutoff)
        else:
            fh = fhp
            fhp = fh.to_absolute(self.cutoff)

        # encode
        X, _ = self._to_dataframe(X, fh=fh)
        # convert
        Xs, _ = self._to_numpy(X, None)

        nfh = int(fh[-1])
        pt = CNNPredictTransform(slots=self._slots)
        ys = pt.fit(self.Xh, self.yh).transform(Xs, fh=nfh)

        for i in range(nfh):
            Xt = pt.step(i)

            yt = self._model.predict(Xt)
            ys[i] = yt[0]
        # end

        yp = self._from_numpy(ys, fhp)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def __repr__(self):
        return f"SimpleCNNForecaster[]"

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# BaseRNNForecaster
# ---------------------------------------------------------------------------

class BaseRNNForecaster(SimpleNNForecaster):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self, *,
                 lags: Union[int, list, tuple, dict] = (0, 1),
                 flavour='lstm',
                 relu=None,
                 activation='relu',
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
                 patience=20,
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
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            callbacks=callbacks,

            flavour=flavour,
            relu=relu,
            activation=activation,
            activation_params=activation_params
        )
        if relu:
            activation = 'relu'

        # self._optimizer = optimizer
        # self._criterion = criterion
        # self._callbacks = callbacks

        # some defaults
        if optimizer is None:
            optimizer = torch.optim.Adam
        if criterion is None:
            criterion = torch.nn.MSELoss
        if patience > 0:
            callbacks = [EarlyStopping(patience=patience)]

        # some classes specified as string
        if isinstance(optimizer, str):
            optimizer = import_from(optimizer)
        if isinstance(criterion, str):
            criterion = import_from(criterion)

        self._lags = lags

        #
        # torchx.nn.LSTM configuration parameters
        #
        # self._flavour = flavour.lower() if isinstance(flavour, str) else flavour

        self._rnn_args = {}
        self._rnn_args['hidden_size'] = hidden_size
        self._rnn_args['num_layers'] = num_layers
        self._rnn_args['bidirectional'] = bidirectional
        self._rnn_args['dropout'] = dropout
        self._rnn_args['activation'] = activation
        self._rnn_args['activation_params'] = activation_params

        #
        # skorch.NeuralNetRegressor configuration parameters
        #
        self._skt_args = {} | kwargs
        self._skt_args["criterion"] = criterion     # replace!
        self._skt_args["optimizer"] = optimizer     # replace!
        self._skt_args["lr"] = lr
        self._skt_args["batch_size"] = batch_size
        self._skt_args["max_epochs"] = max_epochs
        self._skt_args["callbacks"] = callbacks     # replace!

        #
        #
        #

        # self._model = None

        # index
        # self.Xh = None
        # self.yh = None
    # end

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def lags(self):
        return self._lags

    @property
    def flavour(self):
        return self._flavour

    @property
    def hidden_size(self):
        return self._rnn_args['hidden_size']

    @property
    def num_layers(self):
        return self._rnn_args['num_layers']

    @property
    def bidirectional(self):
        return self._rnn_args['bidirectional']

    @property
    def dropout(self):
        return self._rnn_args['dropout']

    @property
    def criterion(self):
        return self._criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr(self):
        return self._skt_args["lr"]

    @property
    def batch_size(self):
        return self._skt_args["batch_size"]

    @property
    def max_epochs(self):
        return self._skt_args["max_epochs"]

    @property
    def callbacks(self):
        return self._callbacks

    @property
    def patience(self):
        return self._skt_args["patience"]

    def get_params(self, deep=True, **kwargs):
        params = {} | self._skt_args | self._rnn_args
        params['lags'] = self._lags
        params['flavour'] = self._flavour

        # convert 'criterion' and 'optimizer' in string
        params['criterion'] = self._criterion
        params['optimizer'] = self._optimizer
        params["callbacks"] = self._callbacks

        return params
    # end

    # -----------------------------------------------------------------------
    # operatiosn
    # -----------------------------------------------------------------------

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
        pass

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        pass

    # -----------------------------------------------------------------------
    # support
    # -----------------------------------------------------------------------

    def _create_skorch_model(self, input_size, output_size):
        pass
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
        # convert X, y to DataFrame
        X, y = self._to_dataframe(X, y)
        # normalize X, y as numpy objects
        Xh, yh = self._to_numpy(X, y)
        # evaluate the input_size/ouput_size
        input_size, output_size = self._compute_input_output_sizes()

        self._model = self._create_skorch_model(input_size, output_size)

        #
        # prepare the data to pass the the Recurrent NN
        #
        tt = RNNTrainTransform(slots=self._slots)
        Xt, yt = tt.fit_transform(Xh, yh)

        self._model.fit(Xt, yt)
        self._save_history()
        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.

        fhp = fh
        if fhp.is_relative:
            fh = fhp
            fhp = fh.to_absolute(self.cutoff)
        else:
            fh = fhp.to_relative(self.cutoff)

        # encode
        X, _ = self._to_dataframe(X, fh=fh)
        # convert
        Xs, _ = self._to_numpy(X, None)

        nfh = int(fh[-1])
        pt = RNNPredictTransform(slots=self._slots)
        ys = pt.fit(self.Xh, self.yh).transform(Xs, fh=nfh)

        for i in range(nfh):
            Xt = pt.step(i)
            yt = self._model.predict(Xt)
            ys[i] = yt[0]
        # end

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
            callbacks__print_log=None,
            **self._skt_args
        )
        # model.set_params(callbacks__print_log=None)

        return model
    # end

    def __repr__(self):
        return f"SimpleRNNForecaster[{self.flavour}]"
# end


# ---------------------------------------------------------------------------
# SlotsRNNForecaster
# LagsRNNForecaster
# ---------------------------------------------------------------------------

class SlotsRNNForecaster(BaseRNNForecaster):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    # def fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
    #     return super().fit(y=y, X=X, fh=fh)

    def _fit(self, y: PD_TYPES, X: PD_TYPES = None, fh: FH_TYPES = None):
        # normalize X, y as 'pandas' objects
        X, y = self._to_dataframe(X, y)
        # normalize X, y as numpy objects
        Xh, yh = self._to_numpy(X, y)
        # evaluate the input_size/ouput_size
        input_size, output_size = self._compute_input_output_sizes()

        self._model = self._create_skorch_model(input_size, output_size)

        #
        # prepare the data to pass the the Recurrent NN
        #
        tt = RNNSlotsTrainTransform(slots=self._slots)
        Xt, yt = tt.fit_transform(Xh, yh)

        self._model.fit(Xt, yt)
        self._save_history()
        return self
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def _predict(self, fh: ForecastingHorizon, X: PD_TYPES = None):
        # [BUG]
        # if X is present and |fh| != |X|, forecaster.predict(fh, X) select the WRONG rows.

        fhp = fh
        if fhp.is_relative:
            fh = fhp
            fhp = fh.to_absolute(self.cutoff)
        else:
            fh = fhp.to_relative(self.cutoff)

        # encode
        X, _ = self._to_dataframe(X, fh=fh)
        # convert
        Xs, _ = self._to_numpy(X)

        nfh = int(fh[-1])
        pt = RNNSlotsPredictTransform(slots=self._slots)
        ys = pt.fit(self.Xh, self.yh).transform(Xs, fh=nfh)

        for i in range(nfh):
            Xt = pt.step(i)
            yt = self._model.predict(Xt)
            ys[i] = yt[0]
        # end

        yp = self._from_numpy(ys, fhp)
        return yp
    # end

    # -----------------------------------------------------------------------
    # Support
    # -----------------------------------------------------------------------

    def _compute_input_output_sizes(self):
        mx = self.Xh.shape[1] if self.Xh is not None else 0
        my = self.yh.shape[1]
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
            callbacks__print_log=None,
            warm_start=True,
            **self._skt_args
        )
        # model.set_params(callbacks__print_log=None)

        return model
    # end

    def __repr__(self):
        return f"SlotsRNNForecaster[{self.flavour}]"
# end


LagsRNNForecaster = SlotsRNNForecaster

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
