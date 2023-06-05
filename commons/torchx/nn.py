import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Union

from torch.utils.data import TensorDataset, DataLoader

from .utils import as_tensor, qualified_name, kwval
from .utils import create_loss_function, create_optimizer, create_layer, LossHistory


# ---------------------------------------------------------------------------
# ConfigurableModule
# ---------------------------------------------------------------------------

class ConfigurableModule(nn.Module):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 layers: list[Union[str, list, dict]],
                 loss: Union[None, str, list, dict] = None,
                 optimizer: Union[None, str, list, dict] = None,
                 data: Union[None, dict] = None,
                 **kwargs):
        super().__init__()
        assert isinstance(layers, list)
        self.layers_config = layers
        self.loss_config = loss
        self.optimizer_config = optimizer
        self.data_config = data if data is not None else {}

        self.module = None

        self._create_layers(kwargs)
        self._configure_data_loader()
        self.loss = create_loss_function(self, self.loss_config)
        self.optimizer = create_optimizer(self, self.optimizer_config)

        self.lh: LossHistory = LossHistory()
    # end

    def _create_layers(self, kwargs):
        layers = []
        for layer_config in self.layers_config:
            layer = create_layer(layer_config, **kwargs)
            layers.append(layer)

        self.module = nn.Sequential(*layers)
    # end

    def _configure_data_loader(self):
        self.batch_size = kwval(self.data_config, 'batch_size', 1)
        self.epochs = kwval(self.data_config, 'epochs', 1)
    # end

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(self, input):
        output = self.module.forward(input)
        return output
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, X, y):
        X = as_tensor(X)
        y = as_tensor(y)
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.train()
        for epoch in range(self.epochs):
            losses = []
            for batch_idx, (X, y) in enumerate(dl):
                y_pred = self(X)
                loss = self.loss(y_pred, y)
                losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # end
            epoch_metric = torch.mean(torch.stack(losses))
            if epoch % 100 == 0:
                print(f"[{epoch}/{self.epochs}] loss: {epoch_metric.item():.5}")
        # end
        return self
    # end

    # -----------------------------------------------------------------------
    # score
    # -----------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray):
        X = as_tensor(X)
        y_true = as_tensor(y)
        ds = TensorDataset(X, y_true)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        errors = []
        for batch, (X, y) in enumerate(dl):
            with torch.no_grad():
                y_pred = self(X)
                error = self._loss(y_pred, y)
                errors.append(error)
        score_metric = torch.mean(torch.stack(errors))
        return score_metric
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def predict(self, X):
        X = as_tensor(X)
        self.eval()
        with torch.no_grad():
            y_pred: torch.Tensor = self(X)
        return y_pred.numpy()
    # end

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class Module(nn.Module):

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(self,
                 module=None,
                 optimizer=None,
                 lr=0.001,
                 criterion=None,
                 max_epochs=1,
                 batch_size=1,
                 log_epochs=1):
        super().__init__()
        if isinstance(module, list):
            self.module = nn.Sequential(*module)
        else:
            self.module = module

        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion()
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        else:
            self.optimizer = optimizer(self.parameters(), lr=lr)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.log_epochs = log_epochs
        self.log = logging.getLogger(qualified_name(self.__class__))
    # end

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(self, input):
        if isinstance(self.module, nn.Sequential):
            models = list(iter(self.module))
            t = input
            for model in models:
                t = model(t)
            output = t
        else:
            output = self.module.forward(input)
        return output
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size=None, epochs=None, val=None) -> "Module":
        if epochs is None: epochs = self.max_epochs
        if batch_size is None: batch_size = self.batch_size

        X = as_tensor(X)
        y = as_tensor(y)
        ds = TensorDataset(X, y)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        if val is not None:
            X_val, y_val = val
            X_val = as_tensor(X_val)
            y_val = as_tensor(y_val)
        # end

        self.train()
        for epoch in range(epochs):
            train_losses = []
            for batch_idx, (X, y) in enumerate(dl):
                y_pred = self(X)
                loss = self.criterion(y_pred, y)
                train_losses.append(loss)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if val is None:
                    continue

            # end

            if epoch % self.log_epochs != 0:
                continue

            if val is None:
                epoch_metric = torch.mean(torch.stack(train_losses))
                print(f"[{epoch}/{self.max_epochs}] loss: {epoch_metric.item():.5}")
            else:
                with torch.no_grad():
                    y_pred = self(X_val)
                    val_loss = self.criterion(y_pred, y_val)

                epoch_metric = torch.mean(torch.stack(train_losses))
                print(f"[{epoch}/{self.max_epochs}] train loss: {epoch_metric.item():.5}, val loss: {val_loss.item()}")

        # end
        return self
    # end

    # -----------------------------------------------------------------------
    # score
    # -----------------------------------------------------------------------

    def score(self, X: np.ndarray, y: np.ndarray):
        X = as_tensor(X)
        y_true = as_tensor(y)
        ds = TensorDataset(X, y_true)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        errors = []
        self.eval()
        for batch, (X, y) in enumerate(dl):
            with torch.no_grad():
                y_pred = self(X)
                error = self._loss(y_pred, y)
                errors.append(error)
        score_metric = torch.mean(torch.stack(errors))
        return score_metric
    # end

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = as_tensor(X)
        self.eval()
        with torch.no_grad():
            y_pred = self(X)
        return y_pred.numpy()
    # end

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


# ---------------------------------------------------------------------------
# RNN/GRU/LSTM
# ---------------------------------------------------------------------------
# It combines in a module a RNN layer connected to a Linear layer,
# batch_first = True for default

#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two LSTMs together to form a `stacked LSTM`,
#             with the second LSTM taking in outputs of the first LSTM and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             LSTM layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
#         proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

class LSTM(nn.LSTM):

    def __init__(self, *, input_size, hidden_size, num_layers=1, output_size=1, batch_first=True, **kwargs):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first= batch_first,
                         **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = {}
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers 
        self.V = nn.Linear(in_features=f*hidden_size, out_features=output_size)

    def forward(self, input, hx=None):
        L = input.shape[0 if self.batch_first else 1]
        D = self.D
        N = self.hidden_size

        if L not in self.hidden:
            hidden_state = torch.zeros(D, L, N, dtype=input.dtype)
            cell_state = torch.zeros(D, L, N, dtype=input.dtype)
            self.hidden[L] = (hidden_state, cell_state)

        hidden = self.hidden[L]
        predict, h = super().forward(input, hidden)
        output = self.V(predict)
        return output
# end

#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two GRUs together to form a `stacked GRU`,
#             with the second GRU taking in outputs of the first GRU and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             GRU layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``
class GRU(nn.GRU):

    def __init__(self, *, input_size, hidden_size, num_layers=1, output_size=1, batch_first=True, **kwargs):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first= batch_first,
                         **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = {}
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers 
        self.V = nn.Linear(in_features=f*hidden_size, out_features=output_size)

    def forward(self, input, hx=None):
        L = input.shape[0 if self.batch_first else 1]
        D = self.D
        N = self.hidden_size

        if L not in self.hidden:
            hidden_state = torch.zeros(D, L, N)
            self.hidden[L] = hidden_state

        hidden = self.hidden[L]
        predict, h = super().forward(input, hidden)
        output = self.V(predict)
        return output
# end


#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two RNNs together to form a `stacked RNN`,
#             with the second RNN taking in outputs of the first RNN and
#             computing the final results. Default: 1
#         nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             RNN layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

class RNN(nn.RNN):

    def __init__(self, *, input_size, hidden_size, num_layers=1, output_size=1, batch_first=True, **kwargs):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first= batch_first,
                         **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden = {}
        f = (2 if self.bidirectional else 1)
        self.D = f*self.num_layers 
        self.V = nn.Linear(in_features=f*hidden_size, out_features=output_size)

    def forward(self, input, hx=None):
        L = input.shape[0 if self.batch_first else 1]
        D = self.D
        N = self.hidden_size

        if L not in self.hidden:
            hidden_state = torch.zeros(D, L, N)
            self.hidden[L] = hidden_state

        hidden = self.hidden[L]
        predict, h = super().forward(input, hidden)
        output = self.V(predict)
        return output
# end


# ---------------------------------------------------------------------------
# DropDimension
# ---------------------------------------------------------------------------

class DropDimensions(nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        if self.dim == -1:
            shape = input.shape[:-2] + torch.Size([-1])
        else:
            raise ValueError("Unsupported index")
        input = input.reshape(shape)
        return input
# end


# ---------------------------------------------------------------------------
# Snake
# ---------------------------------------------------------------------------
# x + 1/f sin(f x)^2
#
# x + (1 - cos(2 f x)/(2 f)

class Snake(nn.Module):

    def __init__(self, frequency=1):
        super().__init__()
        # self.frequency = nn.Parameter(torch.tensor(frequency, dtype=float))
        self.frequency = torch.tensor(frequency, dtype=float)

    def forward(self, x):
        f = self.frequency
        return x + torch.pow(torch.sin(f*x), 2)/f
        # return x + torch.pow(torch.sin(x), 2)


class Difference(nn.Module):

    def __init__(self, n=1):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.diff(x, n=n, dim=-1)

# ---------------------------------------------------------------------------
# PowerModule
# ---------------------------------------------------------------------------
# X^1, X^2, X^2, ...

class PowerModule(nn.Module):

    def __init__(self, order: int = 1, cross: int = 1):
        super().__init__()
        self.order = order
        self.cross = cross

    def forward(self, X):
        if self.order == 1:
            return X
        Xcat = []
        for i in range(1, self.order+1):
            Xi = torch.pow(X, i)
            Xcat.append(Xi)
        return torch.cat(Xcat, 1)
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
