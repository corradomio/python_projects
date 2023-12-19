from typing import Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from torch import Tensor

class RNNComposer:

    def __init__(self, return_sequence, return_state):
        self.return_sequence = return_sequence
        self.return_state = return_state

    def compose_result(self, seq, state):
        if self.return_sequence is True and self.return_state:
            return seq, state
        elif self.return_sequence is True and self.return_state is False:
            return seq
        elif self.return_sequence is False and self.return_state:
            return seq[:, -1], state
        elif self.return_sequence is None and self.return_state:
            return state
        else:
            return seq[:, -1]
# end


# ---------------------------------------------------------------------------
# RNN/GRU/LSTM
# ---------------------------------------------------------------------------
# All torch RNN layers return a tuple composed by
#
#   (sequence, hidden_state)
#
# The following classes extend the original ones adding two parameters to
# decide the result type;
#
#       return_state:       if to return the hidden state
#                           False: no  hidden state is returned
#                           True:  it is returned the hidden state of the last cell
#                           'all': it is returned a tensor with the hidden state of all cells
#       return_sequence:    if to return the sequence, the last value or none
#                           False: it is returned just the last sequence value
#                           True:  it is returned all sequence values
#                           None:  no sequence value is returned
#
# It is changed the default value for 'batch_first': now it is ``True``
#

class LSTM(nn.LSTM):
    """
    Extends nn.LSTM to accept the parameters 'return_sequence' and 'return_state'.
    It changes 'batch_first' to ``True`` as default

    Args:
        input_size
        hidden_size
        num_layers
        bias
        batch_first
        dropout
        bidirectional
        proj_size

        batch_first: Default ``True``
        return_sequence: If ``False`` it returns only the last predicted value
                         if ``True`` it return all predicted values
                         if ``None`` it doesn't return any value
        return_state: If ``False`` it doesn't return the state
                      if ``True`` returns the last state (1, B, Hout)
                      if ``all`` returns all states (1, B, N, Hout)
    """
    # mode: str,
    # input_size: int,
    # hidden_size: int,
    #
    # num_layers: int = 1,
    # bias: bool = True,
    # batch_first: bool = False,
    # dropout: float = 0.,
    # bidirectional: bool = False,
    # proj_size: int = 0,
    # device=None,
    # dtype=None

    def __init__(self,
                 input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.,
                 bidirectional=False,
                 batch_first=True,
                 return_sequence=True,
                 return_state=False,
                 **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            **kwargs
        )
        self.return_sequence = return_sequence
        self.return_state = return_state
        self._composer = RNNComposer(return_sequence, return_state)

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

        if self.return_state == 'all':
            seq, state = self._loop_forward(input, hx)
        else:
            seq, state = super().forward(input, hx)

        # if self.return_sequence is True and self.return_state:
        #     return seq, state
        # elif self.return_sequence is True and self.return_state is False:
        #     return seq
        # elif self.return_sequence is False and self.return_state:
        #     return seq[:, -1], state
        # elif self.return_sequence is None and self.return_state:
        #     return state
        # else:
        #     return seq[:, -1]
        return self._composer.compose_result(seq, state)

    def _loop_forward(self, x, hx):
        n = x.shape[1]
        outs = []
        hs = []
        cs = []
        for i in range(n):
            out, hx = super().forward(x[:, i:i+1], hx)
            outs.append(out)
            h0, c0 = hx
            h0 = h0.unsqueeze(2)
            c0 = c0.unsqueeze(2)
            hs.append(h0)
            cs.append(c0)
        out = torch.cat(outs, dim=1)
        h0 = torch.cat(hs, dim=2)
        c0 = torch.cat(cs, dim=2)
        return out, (h0, c0)
# end


class GRU(nn.GRU):
    """
    Extends nn.GRU to accept the parameters 'return_sequence' and 'return_state'.
    It changes 'batch_first' to ``True`` as default

    Args:
        input_size
        hidden_size
        num_layers
        bias
        batch_first
        dropout
        bidirectional

        batch_first: Default ``True``
        return_sequence: If ``False`` it returns only the last predicted value
                         if ``True`` it return all predicted values
                         if ``None`` it doesn't return any value
        return_state: If ``False`` it doesn't return the state
                      if ``True`` returns the last state
                      if ``all`` returns all states
    """
    def __init__(self,
                 input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.,
                 bidirectional=False,
                 batch_first=True,
                 return_sequence=True,
                 return_state=False,
                 **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            **kwargs
        )
        self.return_sequence = return_sequence
        self.return_state = return_state
        self._composer = RNNComposer(return_sequence, return_state)

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

        if self.return_state == 'all':
            seq, state = self._loop_forward(input, hx)
        else:
            seq, state = super().forward(input, hx)

        # if self.return_sequence is True and self.return_state is True:
        #     return seq, state
        # elif self.return_sequence is True and self.return_state is False:
        #     return seq
        # elif self.return_sequence is False and self.return_state is True:
        #     return seq[:, -1], state
        # elif self.return_sequence is None and self.return_state is True:
        #     return state
        # else:
        #     return seq[:, -1]
        return self._composer.compose_result(seq, state)

    def _loop_forward(self, x, hx):
        n = x.shape[1]
        outs = []
        hs = []
        for i in range(n):
            out, hx = super().forward(x[:, i:i+1], hx)
            outs.append(out)
            h0 = hx.unsqueeze(2)
            hs.append(h0)
        out = torch.cat(outs, dim=1)
        h0 = torch.cat(hs, dim=2)
        return out, h0
# end


class RNN(nn.RNN):
    """
    Extends nn.RNN to accept the parameters 'return_sequence' and 'return_state'.
    It changes 'batch_first' to ``True`` as default

    Args:
        input_size
        hidden_size
        num_layers
        nonlinearity
        bias
        batch_first
        dropout
        bidirectional

        batch_first: Default ``True``
        return_sequence: If ``False`` it returns only the last predicted value
                         if ``True`` it return all predicted values
                         if ``None`` it doesn't return any value
        return_state: If ``False`` it doesn't return the state
                      if ``True`` returns the last state
                      if ``all`` returns all states
    """
    def __init__(self,
                 input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.,
                 bidirectional=False,
                 batch_first=True,
                 return_sequence=True,
                 return_state=False,
                 **kwargs):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            **kwargs
        )
        self.return_sequence = return_sequence
        self.return_state = return_state
        self._composer = RNNComposer(return_sequence, return_state)

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

        if self.return_state == 'all':
            seq, state = self._loop_forward(input, hx)
        else:
            seq, state = super().forward(input, hx)

        # if self.return_sequence is True and self.return_state is True:
        #     return seq, state
        # elif self.return_sequence is True and self.return_state is False:
        #     return seq
        # elif self.return_sequence is False and self.return_state is True:
        #     return seq[:, -1], state
        # elif self.return_sequence is None and self.return_state is True:
        #     return state
        # else:
        #     return seq[:, -1]
        self._composer.compose_result(seq, state)

    def _loop_forward(self, x, hx):
        n = x.shape[1]
        outs = []
        hs = []
        for i in range(n):
            out, hx = super().forward(x[:, i:i+1], hx)
            outs.append(out)
            h0 = hx.unsqueeze(2)
            hs.append(h0)
        out = torch.cat(outs, dim=1)
        h0 = torch.cat(hs, dim=2)
        return out, h0
# end


# ---------------------------------------------------------------------------
# create_rnn
# ---------------------------------------------------------------------------

RNNX_FLAVOURS = {
    'lstm': LSTM,
    'gru': GRU,
    'rnn': RNN,
}

RNNX_PARAMS = [
    'input_size', 'hidden_size', 'num_layers', 'bidirectional', 'bias', 'dropout',
    # extended parameters
    'return_sequence', 'return_state'
]


def create_rnn(flavour: str, **kwargs):
    if flavour not in RNNX_FLAVOURS:
        raise ValueError(f"Unsupported RNN flavour {flavour}")
    else:
        return RNNX_FLAVOURS[flavour](**kwargs)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
