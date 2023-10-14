from typing import Union, Any, Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dim_of(dim: Union[int, list[int]]):
    return [dim] if isinstance(dim, int) else dim


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def is_tuple(v, name, len=None):
    assert isinstance(v, (list, tuple)), f"The parameter {name} must be a tuple list: {v}"
    if len is not None:
        assert len(v) == len, f"The parameter {name} must be a tuple of length {len}: {v}"


# ---------------------------------------------------------------------------
# torch Tensor Utilities
# ---------------------------------------------------------------------------

def to_tensor(v: np.ndarray, dtype=torch.float32) -> Tensor:
    if len(v.shape) == 1:
        v = v.reshape((-1, 1))
    return torch.from_numpy(v).type(dtype)


def time_repeat(x: Tensor, n: int) -> Union[Tensor, list[Tensor]]:
    if isinstance(x, tuple):
        return [time_repeat(t, n) for t in x]
    batch, data_size = x.shape
    r = x.repeat((1, 1, n))
    r = r.reshape((batch, n, data_size))
    return r


def expand_dims(t: Tensor, dim=-1) -> Tensor:
    # if dim == -1:
    #     # return t.reshape(list(t.shape) + [1])
    #     return t[:, None]
    # elif dim == 0:
    #     # return t.reshape([0] + list(t.shape))
    #     return t[None, :]
    # else:
    #     shape = list(t.shape)
    #     return t.reshape(shape[:dim] + [1] + shape[dim:])
    return torch.unsqueeze(t, dim)


def cast(t: Tensor, dtype) -> Tensor:
    return t.to(dtype)


def max(t: Tensor, dim=None, keepdims=False):
    """
    Return the maximum value in the tensor, discarges the index information
    :param t: tensor
    :param dim: axes where to compute the value
    :param keepdims: if to keep all dimensions
    :return: the maximum value
    """
    max_vals, max_idx = torch.max(t, dim=dim, keepdims=keepdims)
    return max_vals


def split(t: Tensor, splits: list[int], dim=1):
    """
    Equivalent to 'tensorflow.split(t, [d0, d1, ...])'
    :param t: tensor to split
    :param splits: width of each dimension
    :return: list of sub-tensors
    """
    assert dim == 1, "It is supported only 'dim=1'"

    if isinstance(splits, int):
        n = t.shape[dim] // splits
        splits = [splits] * n

    parts = []
    i = 0
    for s in splits:
        p = t[:, i:i+s]
        parts.append(p)
        i += s
    return parts


def norm(data, p=1) -> torch.Tensor:
    t = torch.tensor(data)
    if torch.abs(t).sum() == 0:
        return t
    if p == 1:
        t = t / t.sum()
    elif p == 2:
        t = t / torch.sqrt(torch.dot(t, t).sum())
    else:
        t = t / torch.pow(torch.pow(t, p).sum(), 1./p)

    return t


# ---------------------------------------------------------------------------
# NumpyDataset
# ---------------------------------------------------------------------------

# class NumpyDataset(TensorDataset):
#
#     def __init__(self, X, y, dtype=torch.float):
#         assert isinstance(X, np.ndarray)
#         assert isinstance(y, np.ndarray)
#         super().__init__(
#             to_tensor(X, dtype=dtype),
#             to_tensor(y, dtype=dtype)
#         )
# # end


# ---------------------------------------------------------------------------
# LossHistory
# ---------------------------------------------------------------------------

# class LossHistory:
#     def __init__(self):
#         self._train_loss = 0.
#         self._valid_loss = 0.
#         self.train_history = []
#         self.validation_history = []
#         self.iepoch = 0
#
#     def start_epoch(self):
#         self.iepoch += 1
#         self._train_loss = 0.
#         self._valid_loss = 0.
#
#     def train_loss(self, loss):
#         self._train_loss += loss.item()
#         pass
#
#     def end_train(self, n):
#         epoch_loss = self._train_loss / n if n > 0 else 0
#         self.train_history.append(epoch_loss)
#
#     def validation_loss(self, loss):
#         self._valid_loss += loss.item()
#
#     def end_epoch(self, n=0):
#         if n > 0:
#             valid_loss = self._valid_loss / n
#             self.validation_history.append(valid_loss)
#         if self.iepoch %100 > 0:
#             return
#         if len(self.validation_history) > 0:
#             print(f"[{self.iepoch}] train_loss={self.train_history[-1]}, val_loss={self.validation_history[-1]}")
#         else:
#             print(f"[{self.iepoch}] train_loss={self.train_history[-1]}")
#     # end
# # end


# ---------------------------------------------------------------------------
# DataTrainer
# compose_data
# ---------------------------------------------------------------------------
#
#   current = 0x01
#   last    = 0x02
#   input   = 0x03
#   mode    = current | last | input
#
#   (X[-2],y[-2]),       (X[-1],y[-1])      -> y[0]         last=T,current=F
#   (X[-1],y[-1],X[-1]), (X[-1],y[-1],X[0]) -> y[0]         last=T,current=T
#
#   (X[-2],y[-2]),       (X[-1],y[-1])      -> y[-1],y[0]   last=F,current=F
#   (X[-1],y[-1],X[-1]), (X[-1],y[-1],X[0]) -> y[-1],y[0]   last=T,current=T
#

# FLAG_CURRENT = 0x01
# FLAG_LAST    = 0x02
# FLAG_INPUT   = 0x03


# class DataTrainer:
#
#     def __init__(self, slots: list[int], flags: int):
#         if isinstance(slots, int):
#             slots = list(range(slots+1))
#         elif 0 not in slots:
#             slots += [0]
#         slots = list(reversed(slots))
#
#         self.slots = slots
#         self.flags = flags
#         self.X = None
#         self.y = None
#
#     def compose(self, X: Optional[np.ndarray], y: np.ndarray):
#         if len(y.shape) == 1:
#             y = y.reshape((-1, 1))
#
#         assert X is None or isinstance(X, np.ndarray) and len(X.shape) == 2
#         assert isinstance(y, np.ndarray) and len(y.shape) == 2
#
#         self.X = X
#         self.y = y
#
#         Xt = self._compose_xt()
#         yt = self._compose_yt()
#
#         return Xt, yt
#     # end
#
#     def _compose_xt(self):
#         # current:
#         #   False   X[-1],y[-1]
#         #   True    X[-1],y[-1],X[0]
#         #
#         # last
#         #   False   y[-1],y[0]
#         #   True    y[0]
#         #
#         X = self.X
#         y = self.y
#         slots = self.slots
#         current = False if X is None else (self.flags & FLAG_CURRENT != 0)
#
#         s = max(slots)
#         ns = len(slots) - 1
#
#         mx = 0 if X is None else X.shape[1]
#         ny, my = y.shape
#         xdim = mx + my + (mx if current else 0)
#
#         lx = ny - s
#
#         Xt = np.zeros((lx, ns, xdim))
#
#         for i in range(lx):
#             for j in range(ns):
#                 c = slots[j + 0]
#                 if X is not None:
#                     Xt[i, j, 0:mx] = X[s+i-c]
#                 Xt[i, j, mx:mx+my] = y[s+i-c]
#                 if current:
#                     c = slots[j + 1]
#                     Xt[i, j, mx+my:mx+my+mx] = X[s+i-c]
#             # end
#         # end
#
#         return Xt
#     # end
#
#     def _compose_yt(self):
#         # last
#         #   False   y[-1],y[0]
#         #   True    y[0]
#         #
#         y = self.y
#         slots = self.slots
#         last = (self.flags & FLAG_LAST) != 0
#
#         s = max(slots)
#         ns = len(slots) - 1
#
#         ny, my = y.shape
#         ly = ny - s
#
#         if last:
#             yt = np.zeros((ly, 1, my))
#
#             for i in range(ly):
#                 yt[i, 0] = y[s + i]
#         else:
#             yt = np.zeros((ly, ns, my))
#
#             for i in range(ly):
#                 for j in range(ns):
#                     c = slots[j + 1]
#                     yt[i, j] = y[s + i - c]
#
#         return yt
#     # end
# # end


# def compose_data(y: np.ndarray,
#                  X: Optional[np.ndarray] = None,
#                  slots: int = 1,
#                  flags: int = 0) -> tuple[np.ndarray, np.ndarray]:
#
#     dc = DataTrainer(slots, flags=flags)
#     Xt, yt = dc.compose(X, y)
#
#     return Xt, yt
# # end


# ---------------------------------------------------------------------------
# DataPredictor
# predict_recursive
# ---------------------------------------------------------------------------
# Note: to predict y[0], it is necessary to use
#
#   [(X[-t],y[-t])...(X[-1],y[-1])]
#
# then, to predict y[1], it is necessary to use
#
#   [(X[-t+1],y[-t+1])...(X_predict[0],y[0])]

# class DataPredictor:
#
#     def __init__(self, slots: list[int], flags: int = 0):
#         if isinstance(slots, int):
#             slots = list(range(slots+1))
#         elif 0 not in slots:
#             slots += [0]
#         slots = list(reversed(slots))
#
#         self.slots = slots
#         self.flags = flags
#         self.X = None
#         self.y = None
#         self.Xp = None
#         self.yp = None
#         self.Xt = None
#
#     def prepare(self, fh, X, y, Xp):
#         if len(y.shape) == 1:
#             y = y.reshape((-1, 1))
#
#         self.X = X
#         self.y = y
#         self.Xp = Xp
#
#         slots = self.slots
#         current = False if X is None else (self.flags & FLAG_CURRENT != 0)
#
#         s = max(slots)
#         ns = len(slots) - 1
#
#         mx = 0 if X is None else X.shape[1]
#         ny, my = y.shape
#         xdim = mx + my + (mx if current else 0)
#
#         self.Xt = np.zeros((1, ns, xdim))
#         self.yp = np.zeros((fh, my))
#
#         self.yp
#     # end
#
#     def _atx(self, i):
#         return self.X[i] if i < 0 else self.Xp[i]
#
#     def _aty(self, i):
#         return self.y[i] if i < 0 else self.yp[i]
#
#     def compose(self, i):
#         atx = self._atx
#         aty = self._aty
#
#         X = self.X
#         Xt = self.Xt
#
#         slots = self.slots
#         current = False if X is None else (self.flags & FLAG_CURRENT != 0)
#
#         s = max(slots)
#         ns = len(slots) - 1
#         mx = 0 if X is None else X.shape[1]
#         my = self.y.shape[1]
#
#         for j in range(ns):
#             c = slots[j + 0]
#             if X is not None:
#                 Xt[0, j, 0:mx] = atx(i - c)
#             Xt[0, j, mx:mx + my] = aty(i - c)
#             if current:
#                 c = slots[j + 1]
#                 Xt[0, j, mx + my:mx + my + mx] = atx(i - c)
#         # end
#
#         return Xt
#     # end
# # end
#
#
# def prepare_data(fh, X, y, Xp, slots, flags=0):
#     dp = DataPredictor(slots, flags)
#     yp = dp.prepare(fh, X, y, Xp)
#     return dp
#
#
# def predict_recursive(model: nn.Module,
#                       fh: int,
#                       y: np.ndarray,
#                       slots: Union[int, list[int]] = 1,
#                       X: Optional[np.ndarray] = None,
#                       Xp: Optional[np.ndarray] = None,
#                       flags: int = 0):
#
#     dp = DataPredictor(slots, flags)
#     yp = dp.prepare(fh, X, y, Xp)
#
#     for i in range(fh):
#         Xt = dp.compose(i)
#         yt = model.predict(Xt)
#         yp[i] = yt[0]
#     # end
#
#     return yp
# # end


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

# LAYER = "layer"
# OPTIMIZER = "optimizer"
# LOSS = "loss"


# def _normalize_config(config: Union[str, list, tuple, dict], CLASS) -> dict:
#     if isinstance(config, str):
#         config = {CLASS: config}
#     elif isinstance(config, (list, tuple)):
#         if len(config) == 1:
#             config = list(config) + [{}]
#         layer_class = config[0]
#         config = {} | config[1]
#         config[CLASS] = layer_class
#     assert isinstance(config, dict)
#     assert CLASS in config
#     return config
# # end


# def _normalize_class_name(layer_config: dict, CLASS, NS="nn") -> str:
#     a_class: str = layer_config[CLASS]
#     if a_class.startswith("torch." + NS):
#         return a_class
#
#     if a_class.startswith(NS):
#         return "torch." + a_class
#     else:
#         return "torch." + NS + "." + a_class
# # end


# def _class_params(layer_config: dict, CLASS, **kwargs) -> dict:
#     layer_params: dict = {} | layer_config
#     del layer_params[CLASS]
#     for pname in layer_params:
#         pvalue = layer_params[pname]
#         try:
#             pvalue = eval(pvalue, kwargs)
#             layer_params[pname] = pvalue
#         except:
#             pass
#     return layer_params
# # end


# def create_layer(layer_config: Union[str, list, tuple, dict], **kwargs) -> nn.Module:
#     layer_config = _normalize_config(layer_config, LAYER)
#     layer_class_name = _normalize_class_name(layer_config, LAYER, NS="nn")
#     layer_params = _class_params(layer_config, LAYER, **kwargs)
#
#     layer_class = import_from(layer_class_name)
#     layer = layer_class(**layer_params)
#     assert isinstance(layer, nn.Module)
#     return layer
# # end


# def create_optimizer(module: nn.Module, optimizer_config: Union[None, str, list, tuple, dict]) -> torch.optim.Optimizer:
#     if optimizer_config is None:
#         optimizer_config = {
#             OPTIMIZER: "torch.optim.Adam",
#             'lr': 1e-04
#         }
#     optimizer_config = _normalize_config(optimizer_config, OPTIMIZER)
#     optimizer_class_name = _normalize_class_name(optimizer_config, OPTIMIZER, NS="optim")
#     optimizer_params = _class_params(optimizer_config, OPTIMIZER)
#
#     optimizer_class = import_from(optimizer_class_name)
#     optimizer = optimizer_class(module.parameters(), **optimizer_params)
#     assert isinstance(optimizer,  torch.optim.Optimizer)
#     return optimizer
# # end


# def create_loss_function(module: nn.Module, loss_config: Union[None, str, list, tuple, dict]) -> torch.nn.modules.loss._Loss:
#     if loss_config is None:
#         loss_config = {
#             LOSS: "nn.MSELoss"
#         }
#
#     loss_config = _normalize_config(loss_config, LOSS)
#     loss_class_name = _normalize_class_name(loss_config, LOSS, NS="nn")
#     loss_params = _class_params(loss_config, LOSS)
#
#     loss_class = import_from(loss_class_name)
#     loss = loss_class(**loss_params)
#     assert isinstance(loss,  torch.nn.modules.loss._Loss)
#     return loss
#     pass
# # end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
def compose_data():
    return None