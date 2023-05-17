import logging
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from stdlib import import_from, qualified_name, kwval
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


# ---------------------------------------------------------------------------
# NumpyDataset
# ---------------------------------------------------------------------------

def as_tensor(v: np.ndarray) -> torch.Tensor:
    if len(v.shape) == 1:
        v = v.reshape((-1, 1))
    return torch.from_numpy(v).type(torch.float32)
# end


class NumpyDataset(TensorDataset):

    def __init__(self, X, y, dtype=torch.float):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        super().__init__(
            as_tensor(X),
            as_tensor(y)
        )
# end


# ---------------------------------------------------------------------------
# LossHistory
# ---------------------------------------------------------------------------

class LossHistory:
    def __init__(self):
        self._train_loss = 0.
        self._valid_loss = 0.
        self.train_history = []
        self.validation_history = []
        self.iepoch = 0

    def start_epoch(self):
        self.iepoch += 1
        self._train_loss = 0.
        self._valid_loss = 0.

    def train_loss(self, loss):
        self._train_loss += loss.item()
        pass

    def end_train(self, n):
        epoch_loss = self._train_loss / n if n > 0 else 0
        self.train_history.append(epoch_loss)

    def validation_loss(self, loss):
        self._valid_loss += loss.item()

    def end_epoch(self, n=0):
        if n > 0:
            valid_loss = self._valid_loss / n
            self.validation_history.append(valid_loss)
        if self.iepoch %100 > 0:
            return
        if len(self.validation_history) > 0:
            print(f"[{self.iepoch}] train_loss={self.train_history[-1]}, val_loss={self.validation_history[-1]}")
        else:
            print(f"[{self.iepoch}] train_loss={self.train_history[-1]}")
    # end
# end


# ---------------------------------------------------------------------------
# compose_data
# ---------------------------------------------------------------------------

def compose_data(y: np.ndarray, X: Optional[np.ndarray],
                 slots: Union[int, list[int]] = 1,
                 forecast: Union[int, list[int]] = 1) -> tuple[np.ndarray, np.ndarray]:

    if isinstance(slots, int):
        slots = list(range(1, slots+1))
    if isinstance(forecast, int):
        forecast = list(range(forecast))

    n = len(y)
    m = 0 if X is None else X.shape[1]
    s = max(slots)
    t = max(forecast)
    ls = len(slots)
    lf = len(forecast)
    r = n-s-t

    Xt: np.ndarray = np.zeros((r, ls, (m+1)))
    yt: np.ndarray = np.zeros((r, lf))
    rslots = list(reversed(slots))

    for i in range(r):
        for j in range(ls):
            c = rslots[j]
            if X is not None:
                Xt[i, j, 0:m] = X[s+i-c]
            Xt[i, j, m:m+1] = y[s+i-c]
        for j in range(lf):
            c = forecast[j]
            yt[i, j] = y[s+i+c]

    return Xt, yt
# end


# ---------------------------------------------------------------------------
# create_layer
# create_optimizer
# create_loss_function
# ---------------------------------------------------------------------------
# <layer_name>
# nn.<layer_name>
# torch.nn.<layer_name>
#
#   [
#        "<layer_class>",
#       { ...layer_configuration... }
#   ]
#
#   {
#       "layer": "<layer_class>",
#       **layer_configuration
#   }

LAYER = "layer"
OPTIMIZER = "optimizer"
LOSS = "loss"


def _normalize_config(config: Union[str, list, tuple, dict], CLASS) -> dict:
    if isinstance(config, str):
        config = {CLASS: config}
    elif isinstance(config, (list, tuple)):
        if len(config) == 1:
            config = list(config) + [{}]
        layer_class = config[0]
        config = {} | config[1]
        config[CLASS] = layer_class
    assert isinstance(config, dict)
    assert CLASS in config
    return config
# end


def _normalize_class_name(layer_config: dict, CLASS, NS="nn") -> str:
    a_class: str = layer_config[CLASS]
    if a_class.startswith("torch." + NS):
        return a_class

    if a_class.startswith(NS):
        return "torch." + a_class
    else:
        return "torch." + NS + "." + a_class
# end


def _class_params(layer_config: dict, CLASS, **kwargs) -> dict:
    layer_params: dict = {} | layer_config
    del layer_params[CLASS]
    for pname in layer_params:
        pvalue = layer_params[pname]
        try:
            pvalue = eval(pvalue, kwargs)
            layer_params[pname] = pvalue
        except:
            pass
    return layer_params
# end


def create_layer(layer_config: Union[str, list, tuple, dict], **kwargs) -> nn.Module:
    layer_config = _normalize_config(layer_config, LAYER)
    layer_class_name = _normalize_class_name(layer_config, LAYER, NS="nn")
    layer_params = _class_params(layer_config, LAYER, **kwargs)

    layer_class = import_from(layer_class_name)
    layer = layer_class(**layer_params)
    assert isinstance(layer, nn.Module)
    return layer
# end


def create_optimizer(module: nn.Module, optimizer_config: Union[None, str, list, tuple, dict]) -> torch.optim.Optimizer:
    if optimizer_config is None:
        optimizer_config = {
            OPTIMIZER: "torch.optim.Adam",
            'lr': 1e-04
        }
    optimizer_config = _normalize_config(optimizer_config, OPTIMIZER)
    optimizer_class_name = _normalize_class_name(optimizer_config, OPTIMIZER, NS="optim")
    optimizer_params = _class_params(optimizer_config, OPTIMIZER)

    optimizer_class = import_from(optimizer_class_name)
    optimizer = optimizer_class(module.parameters(), **optimizer_params)
    assert isinstance(optimizer,  torch.optim.Optimizer)
    return optimizer
# end


def create_loss_function(module: nn.Module, loss_config: Union[None, str, list, tuple, dict]) -> torch.nn.modules.loss._Loss:
    if loss_config is None:
        loss_config = {
            LOSS: "nn.MSELoss"
        }

    loss_config = _normalize_config(loss_config, LOSS)
    loss_class_name = _normalize_class_name(loss_config, LOSS, NS="nn")
    loss_params = _class_params(loss_config, LOSS)

    loss_class = import_from(loss_class_name)
    loss = loss_class(**loss_params)
    assert isinstance(loss,  torch.nn.modules.loss._Loss)
    return loss
    pass
# end


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

        self.model = None

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

        self.model = nn.Sequential(*layers)
    # end

    def _configure_data_loader(self):
        self.batch_size = kwval(self.data_config, 'batch_size', 1)
        self.epochs = kwval(self.data_config, 'epochs', 1)
    # end

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(self, input):
        output = self.model.forward(input)
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
                 model=None,
                 optimizer=None,
                 loss=None,
                 epochs=1,
                 batch_size=1):
        super().__init__()
        if isinstance(model, list):
            self.model = nn.Sequential(*model)
        else:
            self.model = model

        if loss is None:
            self.loss = nn.MSELoss()
        else:
            self.loss = loss
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.log = logging.getLogger(qualified_name(self.__class__))
    # end

    # -----------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------

    def forward(self, input):
        output = self.model.forward(input)
        return output
    # end

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Module":
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
