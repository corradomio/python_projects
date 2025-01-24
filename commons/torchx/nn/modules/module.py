import torch.nn as nn


class Module(nn.Module):
    """Base class for custom modules"""
    def __init__(self):
        super(Module, self).__init__()
