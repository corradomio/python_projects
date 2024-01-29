import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def positional_encoding(seq_len, d, n=10000, dtype=torch.float32, astensor=True):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i+0] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return torch.tensor(P[np.newaxis, ...], dtype=dtype) if astensor else P
# end


class PositionalEncoder(nn.Module):

    def __init__(self, in_features, max_len=1000, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.max_len = max_len

        # self.pos = torch.zeros((1, max_len, in_features), dtype=dtype)
        # X = (torch.arange(max_len, dtype=dtype).reshape(-1, 1) /
        #      torch.pow(10000, torch.arange(0, in_features, 2, dtype=dtype) / in_features))
        # self.pos[0, :, 0::2] = torch.sin(X)
        # self.pos[0, :, 1::2] = torch.cos(X)

        self.pos = positional_encoding(max_len, in_features, dtype=dtype)

    def forward(self, input):
        seq_len = input.shape[1]
        t = input + self.pos[0, 0:seq_len, :].to(input.device)
        return t
# end
