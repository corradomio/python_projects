import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------
# input:
#   (batch, seq, data)
#
# note that:
#   1. batch is specified during the training
#   2. seq   is not specified, than it is necessary to create data structures
#            for any possible reasonable 'seq' length value. We suppose 1000
#   3. data  is specified by in_features
#
# the output will be as the input:  (batch, seq, data)

class PositionalEncoding(nn.Module):

    def __init__(self, in_features, max_len=100, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.max_len = max_len

        self.pos = torch.zeros((1, max_len, in_features), dtype=dtype)
        X = (torch.arange(max_len, dtype=dtype).reshape(-1, 1) /
             torch.pow(10000, torch.arange(0, in_features, 2, dtype=dtype) / in_features))
        self.pos[0, :, 0::2] = torch.sin(X)
        self.pos[0, :, 1::2] = torch.cos(X)

    def forward(self, input):
        seq_len = input.shape[1]
        t = input + self.pos[0, 0:seq_len, :].to(input.device)
        return t
