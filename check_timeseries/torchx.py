from typing import Union, Optional

import torch
import torch.nn as nn
import numpy as np
from stdlib import import_from

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
            OPTIMIZER: "torch.optim.AdamW",
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
# compose_data
# ---------------------------------------------------------------------------

def compose_data(y: np.ndarray, X: Optional[np.ndarray], slots: Union[int, list[int]]) \
    -> tuple[np.ndarray, np.ndarray]:

    if isinstance(slots, int):
        slots = list(range(1, slots+1))

    n = len(y)
    m = 0 if X is None else X.shape[1]
    s = max(slots)
    ls = len(slots)
    r = n-s

    Xt: np.ndarray = np.zeros((n-s, ls, (m+1)))
    yt: np.ndarray = np.zeros(n-s)
    rslots = list(reversed(slots))

    for i in range(r):
        for j in range(ls):
            c = rslots[j]
            if X is not None:
                Xt[i, j, 0:m] = X[s+i+j-c]
            Xt[i, j, m:m+1] = y[s+i+j-c]
        yt[i] = y[s+i+1]



    return Xt, yt



# ---------------------------------------------------------------------------
# ConfigurableModule
# ---------------------------------------------------------------------------

class ConfigurableModule(nn.Module):

    def __init__(self, layers: list[Union[str, list, dict]], **kwargs):
        super().__init__()
        assert isinstance(layers, list)
        self.layers_config = layers
        self._create_layers(**kwargs)
    # end

    def _create_layers(self, **kwargs):
        layers = []
        for layer_config in self.layers_config:
            layer = create_layer(layer_config, **kwargs)
            layers.append(layer)

        self.model = nn.Sequential(*layers)
    # end

    def forward(self, input):
        output = self.model.forward(input)
        return output
    # end

    def compile(self, optimizer=None, loss=None):
        self._optimizer = create_optimizer(self, optimizer)
        self._loss = create_loss_function(self, loss)
    # end

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
