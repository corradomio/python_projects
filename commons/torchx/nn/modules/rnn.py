from typing import Optional, Tuple, Union, Any

import torch
import torch.nn as nn
from torch import Tensor, matmul, sigmoid, tanh, relu

from stdlib import mul_

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class RNNComposer:

    def __init__(self, return_sequence, return_state, return_shape):
        self.return_sequence = return_sequence
        self.return_state = return_state
        self.return_shape = return_shape

    def compose_result(self, seq, state):
        if seq is not None and not isinstance(self.return_shape, int):
            shape = seq.shape[:-1] + self.return_shape
            seq = torch.reshape(seq, shape)
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


def swapaxes(x, batch_first):
    return torch.swapaxes(x, 0, 1) if batch_first else x


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
#       nonlinearity:       'tanh', 'relu' or None/'identity'
#                           permit to decide the activation function used. For
#                           default is 'tanh' but RNN supports also 'relu'
#                           It is extended to all other RNN types (LSTM & GRU)
#                           Note: it is implemented in python then it is slow!
#
# It is changed the default value for 'batch_first': now it is ``True``
#
# Note: it it is necessary to use 'relu' instead than to use 'GRU' o 'LSTM'
#       it is enough to use 'RNN'
#

class LSTM(nn.LSTM):
    """
    Extends nn.LSTM to accept the parameters 'return_sequence', 'return_state', 'nonlinearity'.
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
        nonlinearity: supported 'tanh' (default), 'relu', None
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
                 input_size: Union[int, tuple[int, ...]],
                 hidden_size: Union[int, tuple[int, ...]],
                 num_layers: int=1,
                 bias: bool=True,
                 dropout: float=0.,
                 bidirectional: bool=False,
                 batch_first: bool=True,
                 return_sequence: bool=True,
                 return_state: bool=False,
                 nonlinearity: str="tanh",
                 **kwargs):
        super().__init__(
            input_size=mul_(input_size),
            hidden_size=mul_(hidden_size),
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            **kwargs
        )

        if nonlinearity == "tanh":
            self.activation = tanh
        elif nonlinearity == "relu":
            self.activation = relu
        else:
            raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

        self.input_shape = input_size
        self.hidden_shape = hidden_size

        self.return_sequence = return_sequence
        self.return_state = return_state
        self.nonlinearity = nonlinearity
        self._use_tanh = (nonlinearity == "tanh")

        self._composer = RNNComposer(return_sequence, return_state, hidden_size)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        t = input
        if len(input.shape) > 3:
            t = torch.flatten(t, start_dim=2)

        # input:
        #   input   (N, L, Hin, ...)
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
            seq, state = self._loop_forward(t, hx)
        else:
            seq, state = self._forward(t, hx)

        return self._composer.compose_result(seq, state)

    def _loop_forward(self, x, hx):
        n = x.shape[1]
        outs = []
        hs = []
        cs = []
        for i in range(n):
            out, hx = self._forward(x[:, i:i+1], hx)
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

    def _forward(self, x, hx):
        if self._use_tanh:
            return super().forward(x, hx)
        else:
            return self._compute(x, hx)

    def _compute(self, x, hx):
        B, S, D = x.shape

        D0 = 0
        D1 = D0 + D
        D2 = D1 + D
        D3 = D2 + D
        D4 = D3 + D

        tanh = self.activation

        # 
        # weight_ih_l[k] – the learnable input-hidden weights of the k^th layer (W_ii|W_if|W_ig|W_io), 
        #   of shape (4*hidden_size, input_size) for k = 0. Otherwise, the shape is     
        #   (4*hidden_size, num_directions * hidden_size). If proj_size > 0 was specified, the shape will be 
        #   (4*hidden_size, num_directions * proj_size) for k > 0
        # 
        # weight_hh_l[k] – the learnable hidden-hidden weights of the k^th layer (W_hi|W_hf|W_hg|W_ho), 
        #   of shape (4*hidden_size, hidden_size). If proj_size > 0 was specified, the shape will be 
        #   (4*hidden_size, proj_size).
        # 
        # bias_ih_l[k] – the learnable input-hidden bias of the k^th layer (b_ii|b_if|b_ig|b_io), 
        #   of shape (4*hidden_size)
        # 
        # bias_hh_l[k] – the learnable hidden-hidden bias of the k^th layer (b_hi|b_hf|b_hg|b_ho), 
        #   of shape (4*hidden_size)
        # 
        # weight_hr_l[k] – the learnable projection weights of the k^th layer 
        #   of shape (proj_size, hidden_size). Only present when proj_size > 0 was specified.
        # 
        # weight_ih_l[k]_reverse – Analogous to weight_ih_l[k] for the reverse direction. 
        #   Only present when bidirectional=True.
        # 
        # weight_hh_l[k]_reverse – Analogous to weight_hh_l[k] for the reverse direction. 
        #   Only present when bidirectional=True.
        # 
        # bias_ih_l[k]_reverse – Analogous to bias_ih_l[k] for the reverse direction. 
        #   Only present when bidirectional=True.
        # 
        # bias_hh_l[k]_reverse – Analogous to bias_hh_l[k] for the reverse direction. 
        #   Only present when bidirectional=True.
        # 
        # weight_hr_l[k]_reverse – Analogous to weight_hr_l[k] for the reverse direction. 
        #   Only present when bidirectional=True and proj_size > 0 was specified.
        #
        if hx is None:
            max_batch_size = B
            num_directions = 2 if self.bidirectional else 1
            hp = torch.zeros(max_batch_size,
                             self.hidden_size,
                             dtype=x.dtype, device=x.device)
            cp = torch.zeros(max_batch_size,
                             self.hidden_size,
                             dtype=x.dtype, device=x.device)
            hx = (hp, cp)

        x = swapaxes(x, self.batch_first)
        hp, cp = hx
        y, hy, cy = [], [], []
        for l in range(self.num_layers):
            y = []
            weight_ih = self.all_weights[l][0]  # W_ir|W_iz|W_in
            weight_hh = self.all_weights[l][1]  # W_hr|W_hz|W_hn
            bias_ih = self.all_weights[l][2]  # b_ir|b_iz|b_in
            bias_hh = self.all_weights[l][3]  # b_hr|b_hz|b_hn

            Wii = weight_ih[D0:D1]
            Wif = weight_ih[D1:D2]
            Wig = weight_ih[D2:D3]
            Wio = weight_ih[D3:D4]

            bii = bias_ih[D0:D1]
            bif = bias_ih[D1:D2]
            big = bias_ih[D2:D3]
            bio = bias_ih[D3:D4]

            Whi = weight_hh[D0:D1]
            Whf = weight_hh[D1:D2]
            Whg = weight_hh[D2:D3]
            Who = weight_hh[D3:D4]

            bhi = bias_hh[D0:D1]
            bhf = bias_hh[D1:D2]
            bhg = bias_hh[D2:D3]
            bho = bias_hh[D3:D4]

            for xt in x:
                it = sigmoid(matmul(xt, Wii) + bii + matmul(hp, Whi) + bhi)
                ft = sigmoid(matmul(xt, Wif) + bif + matmul(hp, Whf) + bhf)
                gt = tanh(matmul(xt, Wig) + big + (matmul(hp, Whg) + bhg))
                ot = sigmoid(matmul(xt, Wio) + bio + matmul(hp, Who) + bho)
                ct = ft*cp + it*gt
                ht = ot*tanh(ct)

                yt = ot[None, ...]
                y.append(yt)
                hp = ht
                cp = ct
            # end
            hy.append(hp[None, ...])
            cy.append(cp[None, ...])
        # end
        y = torch.cat(y, dim=0)
        y = swapaxes(y, self.batch_first)
        hy = torch.cat(hy, dim=0)
        cy = torch.cat(cy, dim=0)
        return y, (hy, cy)
    # end
# end


class GRU(nn.GRU):
    """
    Extends nn.GRU to accept the parameters 'return_sequence', 'return_state', 'nonlinearity'
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
        nonlinearity: supported 'tanh' (default), 'relu', None
    """
    def __init__(self,
                 input_size: Union[int, tuple[int, ...]],
                 hidden_size: Union[int, tuple[int, ...]],
                 num_layers: int=1,
                 bias=True,
                 dropout: float=0.,
                 bidirectional: bool=False,
                 batch_first: bool=True,
                 return_sequence: bool=True,
                 return_state: bool=False,
                 nonlinearity: str="tanh",
                 **kwargs):
        super().__init__(
            input_size=mul_(input_size),
            hidden_size=mul_(hidden_size),
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            # nonlinearity=nonlinearity, NOT SUPPORTED
            **kwargs
        )

        if nonlinearity == "tanh":
            self.activation = tanh
        elif nonlinearity == "relu":
            self.activation = relu
        else:
            raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

        self.input_shape = input_size
        self.hidden_shape = hidden_size

        self.return_sequence = return_sequence
        self.return_state = return_state
        self.nonlinearity = nonlinearity
        self._use_tanh = (nonlinearity == "tanh")

        self._composer = RNNComposer(return_sequence, return_state, hidden_size)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        t = input
        if len(input.shape) > 3:
            t = torch.flatten(t, start_dim=2)

        # input:
        #   input   (N, L, Hin, ...)
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
            seq, state = self._loop_forward(t, hx)
        else:
            seq, state = self._forward(t, hx)

        return self._composer.compose_result(seq, state)

    def _loop_forward(self, x, hx):
        n = x.shape[1]
        outs = []
        hs = []
        for i in range(n):
            out, hx = self._forward(x[:, i:i+1], hx)
            outs.append(out)
            h0 = hx.unsqueeze(2)
            hs.append(h0)
        out = torch.cat(outs, dim=1)
        h0 = torch.cat(hs, dim=2)
        return out, h0

    def _forward(self, x, hx):
        if self._use_tanh:
            return super().forward(x, hx)
        else:
            return self._compute(x, hx)

    def _compute(self, x, hx):
        B, S, D = x.shape

        D0 = 0
        D1 = D0+D
        D2 = D1+D
        D3 = D2+D

        tanh = self.activation

        # weight_ih_l[k] – the learnable input-hidden weights of the k^th layer (W_ir|W_iz|W_in),
        # of shape (3*hidden_size, input_size) for k = 0.
        # Otherwise, the shape is (3*hidden_size, num_directions * hidden_size)
        #
        # weight_hh_l[k] – the learnable hidden-hidden weights of the k^th layer (W_hr|W_hz|W_hn),
        # of shape (3*hidden_size, hidden_size)
        #
        # bias_ih_l[k] – the learnable input-hidden bias of the k^th layer (b_ir|b_iz|b_in),
        # of shape (3*hidden_size)
        #
        # bias_hh_l[k] – the learnable hidden-hidden bias of the k^th layer (b_hr|b_hz|b_hn),
        # of shape (3*hidden_size)
        #
        # num_layers == 1:
        #   x   (32, 12, 16) -> 48 = 16*3
        #   hx  None
        #
        #   weight_ih_l0, weight_hh_l0      (48, 16)
        #   bias_ih_l0,   bias_hh_l0        (48)
        #
        #   all_weights: [weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0]
        if hx is None:
            max_batch_size = B
            num_directions = 2 if self.bidirectional else 1
            hx = torch.zeros(self.num_layers * num_directions,
                             max_batch_size,
                             self.hidden_size,
                             dtype=x.dtype, device=x.device)

        x = swapaxes(x, self.batch_first)
        hp = hx
        y, hy = [], []
        for l in range(self.num_layers):
            y = []
            weight_ih = self.all_weights[l][0]   # W_ir|W_iz|W_in
            weight_hh = self.all_weights[l][1]   # W_hr|W_hz|W_hn
            bias_ih   = self.all_weights[l][2]   # b_ir|b_iz|b_in
            bias_hh   = self.all_weights[l][3]   # b_hr|b_hz|b_hn

            Wir = weight_ih[D0:D1]
            Wiz = weight_ih[D1:D2]
            Win = weight_ih[D2:D3]

            bir = bias_ih[D0:D1]
            biz = bias_ih[D1:D2]
            bin = bias_ih[D2:D3]

            Whr = weight_hh[D0:D1]
            Whz = weight_hh[D1:D2]
            Whn = weight_hh[D2:D3]

            bhr = bias_hh[D0:D1]
            bhz = bias_hh[D1:D2]
            bhn = bias_hh[D2:D3]

            for xt in x:
                rt = sigmoid(matmul(xt, Wir) + bir + matmul(hp, Whr) + bhr)
                zt = sigmoid(matmul(xt, Wiz) + biz + matmul(hp, Whz) + bhz)
                nt = tanh(matmul(xt, Win) + bin + rt*(matmul(hp, Whn) + bhn))
                ht = (1-zt)*nt + zt*hp

                y.append(ht)
                hp = ht
            # end
            hy.append(hp)
        # end
        y = torch.cat(y, dim=0)
        y = swapaxes(y, self.batch_first)
        hy = torch.cat(hy, dim=0)
        return y, hy
    # end
# end


class RNN(nn.RNN):
    """
    Extends nn.RNN to accept the parameters 'return_sequence', 'return_state', 'nonlinearity'
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
        nonlinearity: natively supported 'tanh' and 'relu'
    """
    def __init__(self,
                 input_size: Union[int, tuple[int, ...]],
                 hidden_size: Union[int, tuple[int, ...]],
                 num_layers: int=1,
                 bias: bool=True,
                 dropout: float=0.,
                 bidirectional: bool=False,
                 batch_first: bool=True,
                 return_sequence: bool=True,
                 return_state: bool=False,
                 nonlinearity: str="tanh",
                 **kwargs):
        super().__init__(
            input_size=mul_(input_size),
            hidden_size=mul_(hidden_size),
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            nonlinearity=nonlinearity,
            **kwargs
        )

        if nonlinearity == "tanh":
            self.activation = tanh
        elif nonlinearity == "relu":
            self.activation = relu
        else:
            raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

        self.input_shape = input_size
        self.hidden_shape = hidden_size

        self.return_sequence = return_sequence
        self.return_state = return_state
        self._composer = RNNComposer(return_sequence, return_state, hidden_size)

    def forward(self,
                input: Tensor,
                hx: Optional[Tuple[Tensor, Tensor]] = None) -> Union[Tensor, Tuple[Tensor, Any]]:
        t = input
        if len(input.shape) > 3:
            t = torch.flatten(t, start_dim=2)

        # input:
        #   input   (N, L, Hin, ...)
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
            seq, state = self._loop_forward(t, hx)
        else:
            seq, state = self._forward(t, hx)

        return self._composer.compose_result(seq, state)

    def _loop_forward(self, x, hx):
        n = x.shape[1]
        outs = []
        hs = []
        for i in range(n):
            out, hx = self._forward(x[:, i:i+1], hx)
            outs.append(out)
            h0 = hx.unsqueeze(2)
            hs.append(h0)
        out = torch.cat(outs, dim=1)
        h0 = torch.cat(hs, dim=2)
        return out, h0

    def _forward(self, x, hx):
        # torch RNN supports natively 'nonlinearity' parameter
        return super().forward(x, hx)
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
