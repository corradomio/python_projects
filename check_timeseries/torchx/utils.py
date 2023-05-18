from typing import Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def kwval(d: dict, k: str, v):
    return d[k] if k in d else v


def import_from(qname: str) -> Any:
    """
    Import a class specified by the fully qualified name string

    :param qname: fully qualified name of the class
    :return: Python class
    """
    import importlib
    p = qname.rfind('.')
    qmodule = qname[:p]
    name = qname[p+1:]

    module = importlib.import_module(qmodule)
    clazz = getattr(module, name)
    return clazz
# end


def qualified_name(clazz: type) -> str:
    return f'{clazz.__module__}.{clazz.__name__}'


def as_tensor(v: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    if len(v.shape) == 1:
        v = v.reshape((-1, 1))
    return torch.from_numpy(v).type(dtype)
# end


class NumpyDataset(TensorDataset):

    def __init__(self, X, y, dtype=torch.float):
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        super().__init__(
            as_tensor(X, dtype=dtype),
            as_tensor(y, dtype=dtype)
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

def compose_data(y: np.ndarray,
                 X: Optional[np.ndarray] = None,
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
# End
# ---------------------------------------------------------------------------
