import torch.nn as nn
from torch import pow, sin, cos


class Snake(nn.Module):

    def __init__(self, a=1, method=None):
        super().__init__()
        self.a = float(a)
        self.method = method

    def forward(self, x):
        a = self.a
        if self.method == 'sin':
            return x + 1/a * sin(a*x)
        elif self.method == 'cos':
            return x + 1/a * cos(a*x)
        elif self.method == 'sin2':
            return x + 1/a * pow(sin(a*x), 2)
        else:
            return x + 1/a * sin(a*x)
