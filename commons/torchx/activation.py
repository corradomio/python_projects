import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation function
# ---------------------------------------------------------------------------

NNX_ACTIVATION = {
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


def activation_function(activation, activation_params=None):
    # None     -> None
    # False    -> Identity
    # True     -> ReLU
    # str      -> lookup & create
    # type     -> create
    # instance -> as is
    if activation is None:
        # name = activation
        # activation = NNX_ACTIVATION[name]
        return None

    elif isinstance(activation, bool):
        name = activation
        activation = NNX_ACTIVATION[name]

    elif isinstance(activation, str):
        name = activation.lower()
        activation = NNX_ACTIVATION[name]

    elif issubclass(activation, nn.Module):
        return activation

    if type(activation_params) != dict:
        activation_params = {}

    return activation(**activation_params)
# end
