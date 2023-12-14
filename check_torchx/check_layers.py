import torch.nn as nn


linear = nn.Linear
# y = xA^T + b
# F.linear(input, self.weight, self.bias)


bilinear = nn.Bilinear
# y = x_1^T A x_2 + b


zeropad = nn.ConstantPad1d
# F.pad(input, self.padding, 'constant', self.value)


replicate = nn.ReplicationPad1d
# F.pad(input, self.padding, 'replicate')


attention = nn.MultiheadAttention
# ..

softmax = nn.Softmax
# F.softmax(input, self.dim, _stacklevel=5)

# .
