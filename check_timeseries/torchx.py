from typing import Union

import torch
import torch.nn as nn
from stdlib import import_from

# ---------------------------------------------------------------------------
# create_layer
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


def create_layer(layer_config: Union[str, list, tuple, dict]):
    def _normalize_config(layer_config: Union[str, list, tuple, dict]) -> dict:
        if isinstance(layer_config, str):
            layer_config = {LAYER: layer_config}
        elif isinstance(layer_config, (list, tuple)):
            if len(layer_config) == 1:
                layer_config = list(layer_config) + [{}]
            layer_class = layer_config[0]
            layer_config = {} | layer_config[1]
            layer_config[LAYER] = layer_class
        assert isinstance(layer_config, dict)
        assert LAYER in layer_config
        return layer_config

    def _normalize_layer_class(layer_config: dict) -> str:
        layer_class: str = layer_config[LAYER]
        if layer_class.startswith("torch.nn"):
            return layer_class
        if layer_class.startswith("nn"):
            return "torch." + layer_class
        else:
            return "torch.nn." + layer_class

    def _layer_params(layer_config: dict) -> dict:
        layer_params: dict = {} | layer_config
        del layer_params[LAYER]
        return layer_params

    layer_config = _normalize_config(layer_config)
    layer_class_name = _normalize_layer_class(layer_config)
    layer_params = _layer_params(layer_config)

    layer_class = import_from(layer_class_name)
    return layer_class(**layer_params)
# end


# ---------------------------------------------------------------------------
# PowerModule
# ---------------------------------------------------------------------------

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
