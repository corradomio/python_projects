from typing import Optional, Tuple, Union, Any

import torch.nn as nn
from torch import Tensor


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

class LSTM(nn.LSTM):
    """
    Extends nn.LSTM

    Args:
        batch_first: Default ``True``
        return_sequence: If ``False`` it returns only the last predicted value
        return_state: If ``false`` it doesn't return the state
    """

    def __init__(self,
                 return_sequence=True,
                 return_state=False,
                 batch_first=True,
                 **kwargs):
        super().__init__(batch_first=batch_first, **kwargs)
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
        elif self.return_state:
            return seq[:, -1], (state[0][:, -1], state[1][:, -1])
        else:
            return seq[:, -1]
# end


class GRU(nn.GRU):
    """
    Extends nn.GRU

    Args:
        batch_first: Default ``True``
        return_sequence: If ``False`` it returns only the last predicted value
        return_state: If ``false`` it doesn't return the state
    """
    def __init__(self,
                 return_sequence=True,
                 return_state=False,
                 batch_first=True,
                 **kwargs):
        super().__init__(batch_first=batch_first, **kwargs)
        self.return_sequence = return_sequence
        self.return_state = return_state

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:

        # input:
        #   input   (N, L, Hin)
        #   hx      (D*num_layers, N, Hout)
        # output:
        #   seq     (N, L, D*Hout)
        #   state   (D*num_layers, N, Hout)
        # where:
        #   B:      batch size
        #   N:      sequence length
        #   D:      2 if bidi else 1
        #   Hin:    input_size
        #   Hout:   output_size

        seq, state = super().forward(input, hx)

        if self.return_sequence and self.return_state:
            return seq, state
        elif self.return_sequence:
            return seq
        elif self.return_state:
            return seq[:, -1], state[:, -1]
        else:
            return seq[:, -1]
# end


class RNN(nn.RNN):
    """
    Extends nn.RNN

    Args:
        batch_first: Default ``True``
        return_sequence: If ``False`` it returns only the last predicted value
        return_state: If ``false`` it doesn't return the state
    """
    def __init__(self,
                 return_sequence=True,
                 return_state=False,
                 batch_first=True,
                 **kwargs):
        super().__init__(batch_first=batch_first, **kwargs)
        self.return_sequence = return_sequence
        self.return_state = return_state

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:

        # input:
        #   input   (N, L, Hin)
        #   hx      (D*num_layers, N, Hout)
        # output:
        #   seq     (N, L, D*Hout)
        #   state   (D*num_layers, N, Hout)
        # where:
        #   B:      batch size
        #   N:      sequence length
        #   D:      2 if bidi else 1
        #   Hin:    input_size
        #   Hout:   output_size

        seq, state = super().forward(input, hx)

        if self.return_sequence and self.return_state:
            return seq, state
        elif self.return_sequence:
            return seq
        elif self.return_state:
            return seq[:, -1], state[:, -1]
        else:
            return seq[:, -1]
# end


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RNNX_FLAVOURS = {
    'lstm': LSTM,
    'gru': GRU,
    'rnn': RNN,
}

RNNX_PARAMS = [
    'input_size', 'hidden_size', 'num_layers', 'bidirectional', 'bias', 'dropout',
    'return_sequence', 'return_state'
]
