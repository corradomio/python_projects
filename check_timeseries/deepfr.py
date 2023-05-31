# from typing import Union, Optional
#
# import numpy as np
# import numpyx as npx
# import pandas as pd
# import pandasx as pdx
# import skorch
# import sktimex as sktx
# import stdlib
# import torch.optim
# import torchx.nn as nnx
# from sklearn.preprocessing import MinMaxScaler
# from skorchx.callbacks import EarlyStopping
# from sktimex.utils import PD_TYPES, FH_TYPES
#
# # nnx.LSTM
# #   input_size      this depends on lagx, |X[0]| and |y[0]|
# #   hidden_size     2*input_size
# #   output_size=1
# #   num_layers=1
# #   bias=True
# #   batch_first=True
# #   dropout=0
# #   bidirectional=False
# #   proj_size =0
# #
# # Note: the neural netword can be created ONLY during 'fit', because the NN structure
# # depends on the configuration AND X, y dimensions/ranks
# #
#
# RNN_FLAVOURS = {
#     'lstm': nnx.LSTM,
#     'gru': nnx.GRU,
#     'rnn': nnx.RNN,
#     'LSTM': nnx.LSTM,
#     'GRU': nnx.GRU,
#     'RNN': nnx.RNN
#
# }
#
#
# class NeuralNetForecaster:
#
#     def __init__(self, *,
#                  lag: Union[int, list, tuple, dict] = (0, 1),
#                  current: Optional[bool] = None,
#                  target: Optional[str] = None,
#                  y_only: bool = False,
#                  periodic: Union[None, str, tuple] = None,
#                  scale: bool = False,
#
#                  steps: int = 1,
#
#                  hidden_size: int = 1,
#                  num_layers: int = 1,
#                  bidirectional: bool = False,
#                  dropout: float = 0,
#
#                  flavour: str = 'lstm',
#                  criterion=None,
#                  optimizer=None,
#                  lr=0.01,
#                  batch_size: int = 16,
#                  max_epochs: int = 300,
#                  callbacks=None,
#                  patience=20,
#                  **kwargs
#                  ):
#         """
#
#         :param lag: input/target lags
#         :param current: if to use the current slot (used when lag is a integer)
#         :param target: name of the target column
#         :param y_only: if to use target only
#         :param period: if to add periodic information
#         :param steps: length of the sequence length
#         :param flavour: type of RNN ('lstm', 'gru', 'rnn')
#         :param optimizer: class of the optimizer to use (default: Adam)
#         :param criterion: class of the loss to use (default: MSLoss)
#         :param batch_size: back size (default 16)
#         :param max_epochs: EPOCHS (default 300)
#         :param hidden_size: number of RNN hidden layers
#         :param num_layers: number of RNN layers
#         :param bidirectional: if to use a bidirectional
#         :param dropout: if to apply a dropout
#         :param kwargs: other parameters
#         """
#         super().__init__()
#
#         # some defaults
#         if optimizer is None:
#             optimizer = torch.optim.Adam
#         if criterion is None:
#             criterion = torch.nn.MSELoss
#         if patience > 0:
#             callbacks = [EarlyStopping(patience=patience)]
#
#         # some classes specified as string
#         if isinstance(optimizer, str):
#             optimizer = stdlib.import_from(optimizer)
#         if isinstance(criterion, str):
#             criterion = stdlib.import_from(criterion)
#
#         self._lag = lag
#         self._target = target
#         self._current = current
#         self._y_only = y_only
#         self._periodic = periodic
#         self._scale = scale
#
#         self._flavour = flavour
#         self._steps = steps
#
#         #
#         # torchx.nn.LSTM configuration parameters
#         #
#         self._rnn_args = {}
#         self._rnn_args['hidden_size'] = hidden_size
#         self._rnn_args['num_layers'] = num_layers
#         self._rnn_args['bidirectional'] = bidirectional
#         self._rnn_args['dropout'] = dropout
#
#         #
#         # skorch.NeuralNetRegressor configuration parameters
#         #
#         self._skt_args = {} | kwargs
#         self._skt_args["criterion"] = criterion
#         self._skt_args["optimizer"] = optimizer
#         self._skt_args["lr"] = lr
#         self._skt_args["batch_size"] = batch_size
#         self._skt_args["max_epochs"] = max_epochs
#         self._skt_args["callbacks"] = callbacks
#
#         lags = sktx.resolve_lag(lag, current)
#
#         self._xlags = lags.input
#         self._ylags = lags.target
#         self._x_scaler = MinMaxScaler()
#         self._y_scaler = MinMaxScaler()
#         self._model = None
#
#         # index
#         self.Ih = None
#         self.Xh = None
#         self.yh = None
#     # end
#
#     # -----------------------------------------------------------------------
#     # Properties
#     # -----------------------------------------------------------------------
#
#     def get_params(self, deep=True, **kwargs):
#         params = {} | self._skt_args | self._rnn_args
#         params['lag'] = self._lag
#         params['target'] = self._target
#         params['y_only'] = self._y_only
#         params['flavour'] = self._flavour
#         params['steps'] = self._steps
#
#         # convert 'criterion' and 'optimizer' in string
#         params['criterion'] = str(params['criterion'])
#         params['optimizer'] = str(params['optimizer'])
#
#         return params
#     # end
#
#     # -----------------------------------------------------------------------
#     # fit
#     # -----------------------------------------------------------------------
#
#     def _to_dataframe(self, X, y, fh=None):
#         if y is not None:
#             self.Ih = y.index
#
#         if X is None:
#             if y is not None:
#                 X = pd.DataFrame({}, index=y.index)
#             else:
#                 cutoff = self.Ih[-1]
#                 index = pd.period_range(cutoff+1, periods=fh)
#                 X = pd.DataFrame({}, index=index)
#         # end
#
#         if isinstance(X, pd.Series):
#             X = pd.DataFrame({"X": X}, index=X.index)
#         if isinstance(y, pd.Series):
#             y = pd.DataFrame({"y": X}, index=y.index)
#         return X, y
#
#     def _encode_periodic(self, X):
#         if self._periodic:
#             X = pdx.periodic_encode(X)
#         return X
#
#     def _to_numpy(self, X, y):
#         Xs = X.to_numpy().astype(np.float32)
#         if y is not None:
#             ys = y.to_numpy().astype(np.float32)
#         else:
#             ys = None
#
#         if len(Xs.shape) == 1:
#             Xs = Xs.reshape((-1, 1))
#         if ys is not None and len(ys.shape) == 1:
#             ys = ys.reshape((-1, 1))
#
#         if not self._scale:
#             return Xs, ys
#
#         if ys is not None:
#             Xs = self._x_scaler.fit_transform(Xs).astype(np.float32)
#             ys = self._y_scaler.fit_transform(ys).astype(np.float32)
#             self.Xh = Xs
#             self.yh = ys
#         else:
#             Xs = self._x_scaler.transform(X).astype(np.float32)
#             ys = None
#
#         return Xs, ys
#     # end
#
#     def _from_numpy(self, ys):
#         if self._scale:
#             ys = self._y_scaler.inverse_transform(ys).astype(self.yh.dtype)
#
#         # 1D array
#         ys = ys.reshape(-1)
#         n = len(ys)
#         cutoff = self.Ih[-1]
#         index = pd.period_range(cutoff+1, periods=n)
#         return pd.Series(ys, index=index)
#     # end
#
#     def _evaluate_input_output_sizes(self):
#         xlags = self._xlags
#         ylags = self._ylags
#         mx = self.Xh.shape[1]
#         my = self.yh.shape[1]
#         input_size = mx * len(xlags) + my * len(ylags)
#         return input_size, my
#
#     def fit(self, y: PD_TYPES, X: PD_TYPES=None):
#         if self._y_only:
#             X = None
#
#         # normalize X, y as 'pandas' objects
#         X, y = self._to_dataframe(X, y)
#         # encode periodic data
#         X = self._encode_periodic(X)
#         # normalize X, y as numpy objects
#         Xh, yh = self._to_numpy(X, y)
#         # evaluate the input_size/ouput_size
#         input_size, output_size = self._evaluate_input_output_sizes()
#
#         # create the torch model
#         #   input_size      this depends on lagx, |X[0]| and |y[0]|
#         #   hidden_size     2*input_size
#         #   output_size=1
#         #   num_layers=1
#         #   bias=True
#         #   batch_first=True
#         #   dropout=0
#         #   bidirectional=False
#         #   proj_size =0
#         rnn_constructor = RNN_FLAVOURS[self._flavour]
#         rnn = rnn_constructor(
#                     input_size=input_size,
#                     output_size=output_size,
#                     **self._rnn_args)
#
#         # create the skorch model
#         #   module: torch module
#         #   criterion:  loss function
#         #   optimizer: optimizer
#         #   lr
#         #
#         self._model = skorch.NeuralNetRegressor(
#             module=rnn,
#             **self._skt_args
#         )
#
#         #
#         # prepare the data to pass the the Recurrent NN
#         #
#         lu = npx.UnfoldLoop(self._steps, xlags=self._xlags, ylags=self._ylags)
#         Xt, yt = lu.fit_transform(Xh, yh)
#
#         self._model.fit(Xt, yt)
#         pass
#
#     def predict(self, fh=None, X=None):
#         if self._y_only:
#             X = None
#
#         if fh is None and X is not None:
#             fh = len(X)
#         # encode
#         X, _ = self._to_dataframe(X, None, fh=fh)
#         # encode periodic data
#         X = self._encode_periodic(X)
#         # convert
#         Xs, _ = self._to_numpy(X, None)
#
#         up = npx.UnfoldPreparer(self._steps, xlags=self._xlags, ylags=self._ylags)
#         ys = up.fit(self.Xh, self.yh).transform(Xs, fh)
#
#         for i in range(fh):
#             Xt = up.step(i)
#             yt = self._model.predict(Xt)
#             ys[i] = yt[0, -1]
#         # end
#
#         yp = self._from_numpy(ys)
#         return yp
#     # end
# # end
