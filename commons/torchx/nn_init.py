import torch.nn


NN_INIT_METHODS = {

    "uniform": torch.nn.init.uniform_,
    "normal": torch.nn.init.normal_,
    "xavier_uniform": torch.nn.init.xavier_uniform_,
    "xavier_normal": torch.nn.init.xavier_normal_,
    "kaiming_uniform": torch.nn.init.kaiming_uniform_,
    "kaiming_normal": torch.nn.init.kaiming_normal_,

    "constant": torch.nn.init.constant_,
    "ones": torch.nn.init.ones_,
    "zeros": torch.nn.init.zeros_,
    "eye": torch.nn.init.eye_,
    "dirac": torch.nn.init.dirac_,
    "orthogonal": torch.nn.init.orthogonal_,
    "sparse": torch.nn.init.sparse_,

}


def tensor_initialize(tensor, init_method, init_params=None):
    """Initialize the module based on the specified initialization method"""
    if init_params is None:
        init_params = {}
    if isinstance(init_method, str):
        init_method = NN_INIT_METHODS[init_method]
    if isinstance(init_method, type):
        init_method = init_method(init_params)
    return init_method(tensor)

