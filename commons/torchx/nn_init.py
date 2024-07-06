from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim

from stdlib import import_from

# ---------------------------------------------------------------------------
# Initialization methods
# ---------------------------------------------------------------------------

NN_INIT_METHODS = {
    "uniform": nn.init.uniform_,
    "normal": nn.init.normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "xavier_normal": nn.init.xavier_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,

    "constant": nn.init.constant_,
    "ones": nn.init.ones_,
    "zeros": nn.init.zeros_,
    "eye": nn.init.eye_,
    "dirac": nn.init.dirac_,
    "orthogonal": nn.init.orthogonal_,
    "sparse": nn.init.sparse_,
}

# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------

NNX_ACTIVATION_FUNCTIONS = {
    None: nn.Identity,
    False: nn.Identity,
    True: nn.ReLU,

    "linear": nn.Identity,
    "identity": nn.Identity,

    "relu": nn.ReLU,
    "elu": nn.ELU,
    "hardshrink": nn.Hardshrink,
    "hardsigmoid": nn.Hardsigmoid,
    "hardtanh": nn.Hardtanh,
    "hardswish": nn.Hardswish,
    "leakyrelu": nn.LeakyReLU,
    "logsigmoid": nn.LogSigmoid,
    "multiheadattention": nn.MultiheadAttention,
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "rrelu": nn.RReLU,
    "selu": nn.SELU,
    "celu": nn.CELU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "softplus": nn.Softplus,
    "softshrink": nn.Softshrink,
    "softsign": nn.Softsign,
    "tanh": nn.Tanh,
    "tanhshrink": nn.Tanhshrink,
    "threshold": nn.Threshold,
    "glu": nn.GLU,
    "softmin": nn.Softmin,
    "softmax": nn.Softmax,
    "softmax2d": nn.Softmax2d,
    "logsoftmax": nn.LogSoftmax,
    "adaptivelogsoftmaxwithloss": nn.AdaptiveLogSoftmaxWithLoss
}


def select_activation(activation) -> type:

    if isinstance(activation, type):
        pass
    elif activation in NNX_ACTIVATION_FUNCTIONS:
        activation = NNX_ACTIVATION_FUNCTIONS[activation]
    elif isinstance(activation, str):
        activation = import_from(activation)
    elif issubclass(activation, nn.Module):
       pass
    return activation


def activation_function(activation, kwargs=None):
    # None     -> None
    # False    -> Identity
    # True     -> ReLU
    # str      -> lookup & create
    # type     -> create
    # instance -> as is

    if kwargs is None: kwargs = {}
    activation_fun = select_activation(activation)
    return activation_fun(**kwargs)
# end


# ---------------------------------------------------------------------------
# lr_scheduler functions
# ---------------------------------------------------------------------------

NNX_LR_SCHEDULERS = {
    "lambdalr": optim.lr_scheduler.LambdaLR,
    "multiplicativelr": optim.lr_scheduler.MultiplicativeLR,
    "steplr": optim.lr_scheduler.StepLR,
    "multisteplr": optim.lr_scheduler.MultiStepLR,
    "constantlr": optim.lr_scheduler.ConstantLR,
    "linearlr": optim.lr_scheduler.LambdaLR,
    "exponentiallr": optim.lr_scheduler.ExponentialLR,
    "sequentiallr": optim.lr_scheduler.ReduceLROnPlateau,
    "polynomiallr": optim.lr_scheduler.PolynomialLR,
    "cosineannealinglr": optim.lr_scheduler.CosineAnnealingLR,
    "chainedscheduler": optim.lr_scheduler.ChainedScheduler,
    "reducelronplateau": optim.lr_scheduler.ReduceLROnPlateau,
    "cycliclr": optim.lr_scheduler.CyclicLR,
    "cosineannealingwarmrestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "onecycclelr": optim.lr_scheduler.OneCycleLR,
}


def select_lr_scheduler(lr_scheduler: Union[None, str, type]):
    if lr_scheduler is None:
        return torch.optim.lr_scheduler.StepLR
    elif isinstance(lr_scheduler, type):
        return lr_scheduler
    elif isinstance(lr_scheduler, str) and lr_scheduler in NNX_LOSS_FUNCTIONS:
        return NNX_LR_SCHEDULERS[lr_scheduler]
    elif isinstance(lr_scheduler, str):
        return import_from(lr_scheduler)
    else:
        raise ValueError(f"Unsupported criterion {lr_scheduler}")


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

NNX_LOSS_FUNCTIONS = {
    'l1': nn.L1Loss,
    'mse': nn.MSELoss,
    'crossentropy': nn.CrossEntropyLoss,
    'ctc': nn.CTCLoss,
    'nll': nn.NLLLoss,
    'poisson': nn.PoissonNLLLoss,
    'gaussiannll': nn.GaussianNLLLoss,
    'kldiv': nn.KLDivLoss,
    'bce': nn.BCELoss,
    'bcewithlogits': nn.BCEWithLogitsLoss,
    'marginranking': nn.MarginRankingLoss,
    'hingeembedding': nn.HingeEmbeddingLoss,
    'multilabelmargin': nn.MultiLabelMarginLoss,
    'huber': nn.HuberLoss,
    'smoothl1': nn.SmoothL1Loss,
    'softmargin': nn.SoftMarginLoss,
    'multilabelsoftmargin': nn.MultiLabelSoftMarginLoss,
    'cosineembedding': nn.CosineEmbeddingLoss,
    'multimargin': nn.MultiMarginLoss,
    'tripletmargin': nn.TripletMarginLoss,
    'tripletmarginwithdistance': nn.TripletMarginWithDistanceLoss,
}


def select_loss(loss: Union[None, str, type]):
    if loss is None:
        return torch.nn.MSELoss
    elif isinstance(loss, str) and loss in NNX_LOSS_FUNCTIONS:
        return NNX_LOSS_FUNCTIONS[loss]
    elif isinstance(loss, str):
        return import_from(loss)
    elif isinstance(loss, type):
        return loss
    else:
        raise ValueError(f"Unsupported criterion {loss}")


select_criterion = select_loss


def loss_fun(loss: Union[None, str, type], kwargs: dict):
    if kwargs is None: kwargs = {}
    loss_class = select_loss(loss)
    return loss_class(**kwargs)


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

NNX_OPTIMIZERS = {
    'adadelta': optim.Adadelta,
    'adagrad': optim.Adagrad,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
    'sparseadam': optim.SparseAdam,
    'adamax': optim.Adamax,
    'asgd': optim.ASGD,
    'lbfgs': optim.LBFGS,
    'nadam': optim.NAdam,
    'radam': optim.RAdam,
    'rmsprop': optim.RMSprop,
    'rprop': optim.Rprop,
    'sdg': optim.SGD,
}


def select_optimizer(optim: Union[None, str, type]):
    if optim is None:
        return torch.optim.Adam
    elif isinstance(optim, type):
        return optim
    elif isinstance(optim, str) and optim in NNX_OPTIMIZERS:
        return NNX_OPTIMIZERS[optim]
    elif isinstance(optim, str):
        return import_from(optim)
    else:
        raise ValueError(f"Unsupported optimizer {optim}")


def optimizer(optim: Union[None, str, type], kwargs: dict):
    if kwargs is None: kwargs = {}
    optim_class = select_optimizer(optim)
    return optim_class(**kwargs)


# ---------------------------------------------------------------------------
# tensor_initialize
# ---------------------------------------------------------------------------

def tensor_initialize(tensor, init_method, init_params=None):
    """Initialize the module based on the specified initialization method"""
    if init_params is None:
        init_params = {}
    if isinstance(init_method, type):
        init_method = init_method(init_params)
    if isinstance(init_method, str):
        init_method = NN_INIT_METHODS[init_method]
    return init_method(tensor)


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------

