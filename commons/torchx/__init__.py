import torch
import logging

from .nn_init import NN_INIT_METHODS, NNX_OPTIMIZERS, NNX_LOSS_FUNCTIONS

from .nnlin import GRULinear, LSTMLinear, RNNLinear
from .utils import print_shape, expand_dims, cast, split, max, time_repeat
from .nn_init import select_optimizer, select_criterion, select_loss, select_lr_scheduler
from .nn_init import select_activation, select_criterion, select_loss, select_lr_scheduler, select_optimizer
from .nn_init import activation_function

orig_torch_load = torch.load

def torch_wrapper(*args, **kwargs):
    logging.warning("[comfyui-unsafe-torch] I have unsafely patched `torch.load`.  The `weights_only` option of `torch.load` is forcibly disabled.")
    kwargs['weights_only'] = False

    return orig_torch_load(*args, **kwargs)

torch.load = torch_wrapper

NODE_CLASS_MAPPINGS = {}
__all__ = ['NODE_CLASS_MAPPINGS']

