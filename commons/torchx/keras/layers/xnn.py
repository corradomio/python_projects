import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Repeat
# ---------------------------------------------------------------------------
# As TF RepeatedVector
#

class RepeatVector(nn.Module):

    def __init__(self, n_repeat=1):
        super().__init__()
        self.n_repeat = n_repeat

    def forward(self, input: torch.Tensor):
        rep_list = [input for i in range(self.n_repeat)]
        repeated = torch.stack(rep_list, 1)
        return repeated
# end


# ---------------------------------------------------------------------------
# TimeDistributed
# ---------------------------------------------------------------------------
# As TF TimeDistributed
#
#   LSTM:           (batch, seq, 1    )   ->  (batch, seq, units)
#   Distributed:    (batch, seq, units)   ->  (batch, seq,     1)
#

class TimeDistributed(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        n_repeat = input.shape[1]

        y_list = []
        for i in range(n_repeat):
            y = self.model.forward(input[:, i])
            y_list.append(y)

        return torch.cat(y_list, dim=1)
# end
