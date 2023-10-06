from typing import Optional, Tuple, Union, Any

import torch.nn as nn
from torch import Tensor

from ..utils import TorchLayerMixin


# ---------------------------------------------------------------------------
# RNN/GRU/LSTM
# ---------------------------------------------------------------------------
# All torch RNN layers return a tuple composed by
#
#   (sequence, hidden_state)
#
# these classes extend the original implementation adding two parameters to
# decide the result type;
#
#       return_state:       if to return the hidden state
#       return_sequence:    if to return the sequenze or just the last value

class LSTM(nn.LSTM, TorchLayerMixin):

    def __init__(self,
                 return_sequence=True,
                 return_state=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_sequence = return_sequence
        self.return_state = return_state

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        seq, state = super().forward(input, hx)

        if self.return_sequence and self.return_state:
            return seq, state
        elif self.return_sequence:
            return seq
        else:
            return seq[:, -1]
# end


class GRU(nn.GRU, TorchLayerMixin):

    def __init__(self,
                 return_sequence=True,
                 return_state=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_sequence = return_sequence
        self.return_state = return_state

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        seq, state = super().forward(input, hx)

        if self.return_sequence and self.return_state:
            return seq, state
        elif self.return_sequence:
            return seq
        else:
            return seq[:, -1]
# end


class RNN(nn.RNN, TorchLayerMixin):

    def __init__(self,
                 return_sequence=True,
                 return_state=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_sequence = return_sequence
        self.return_state = return_state

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        seq, state = super().forward(input, hx)

        if self.return_sequence and self.return_state:
            return seq, state
        elif self.return_sequence:
            return seq
        else:
            return seq[:, -1]
# end
