import torch
a = torch.rand(10, requires_grad=True)
b = torch.rand(10, requires_grad=True)
scalar = (a + b).sum()

