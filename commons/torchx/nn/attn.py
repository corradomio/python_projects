import torch
import torch.nn as nn
from torch import Tensor
from ..activation import activation_function
from ..utils import TorchLayerMixin
from ..utils import expand_dims, cast, max
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

# ---------------------------------------------------------------------------
# SelfAttention
# ---------------------------------------------------------------------------

class SelfAttention(nn.MultiheadAttention, TorchLayerMixin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        return super().forward(input, input, input)


# ---------------------------------------------------------------------------
# SequentialSelfAttention
# ---------------------------------------------------------------------------
#
# keras.layers.SeqSelfAttention
#       units=32,
#       attention_width=None,
#       attention_type=ATTENTION_TYPE_ADD,
#       return_attention=False,
#       history_only=False,
#       kernel_initializer='glorot_normal',
#       bias_initializer='zeros',
#       kernel_regularizer=None,
#       bias_regularizer=None,
#       kernel_constraint=None,
#       bias_constraint=None,
#       use_additive_bias=True,
#       use_attention_bias=True,
#       attention_activation=None,
#       attention_regularizer_weight=0.0,
#       **kwargs.

# torch.nn.MultiHeadAttention
#       embed_dim,
#       num_heads,
#       dropout=0.,
#       bias=True,
#       add_bias_kv=False,
#       add_zero_attn=False,
#       dim=None,
#       vdim=None,
#       batch_first=False,
#       device=None,
#       dtype=None

class SequentialSelfAttention(nn.Module, TorchLayerMixin):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self, in_features, out_features,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 attention_activation=None,
                 bias=True,
                 attention_bias=True,
                 return_attention=False,
                 history_only=False):
        super().__init__()
        self.in_features = in_features              # input
        self.out_features = out_features            # units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.attention_activation = attention_activation
        self.return_attention = return_attention
        self.history_only = history_only
        self.bias = bias
        self.attention_bias = attention_bias

        if history_only and attention_width is None:
            self.attention_width = int(1e9)
        if attention_type == self.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == self.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

        self._build_parameters()

        self._attention_activation = activation_function(self.attention_activation)

        self._reset_parameters()
    # end

    def _build_parameters(self):
        feature_dim = self.in_features
        units = self.out_features
        attention_type = self.attention_type

        if attention_type == self.ATTENTION_TYPE_ADD:
            self.Wt = nn.Parameter(torch.zeros(feature_dim, units))
            self.Wx = nn.Parameter(torch.zeros(feature_dim, units))
            self.Wa = nn.Parameter(torch.zeros(units, 1))
            if self.bias:
                self.bh = nn.Parameter(torch.zeros(units))
            if self.attention_bias:
                self.ba = nn.Parameter(torch.zeros(1))
        elif attention_type == self.ATTENTION_TYPE_MUL:
            self.Wa = nn.Parameter(torch.zeros(feature_dim, feature_dim))
            if self.attention_bias:
                self.ba = nn.Parameter(torch.zeros(1))
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def _reset_parameters(self):
        attention_type = self.attention_type

        if attention_type == self.ATTENTION_TYPE_ADD:
            xavier_uniform_(self.Wt)
            xavier_uniform_(self.Wx)
            xavier_uniform_(self.Wa)
            if self.bias:
                xavier_uniform_(self.bh)
            if self.attention_bias:
                xavier_uniform_(self.ba)
        elif attention_type == self.ATTENTION_TYPE_MUL:
            xavier_uniform_(self.Wa)
            if self.attention_bias:
                xavier_uniform_(self.ba)
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def forward(self, inputs: Tensor) -> Tensor:
        input_len = inputs.shape[1]
        dtype = inputs.dtype

        if self.attention_type == self.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        else:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self._attention_activation(e)

        if self.attention_width is not None:
            if self.history_only:
                lower: Tensor = torch.arange(0, input_len) - (self.attention_width - 1)
            else:
                lower: Tensor = torch.arange(0, input_len) - self.attention_width // 2
            lower = expand_dims(lower, -1)
            upper = lower + self.attention_width
            indices = expand_dims(torch.arange(0, input_len), dim=0)
            p = 10000.0 * (1.0 - cast(lower <= indices, dtype) * cast(indices < upper, dtype))
            e -= p

        # support for masking not implemented

        # a_t = \text{softmax}(e_t)
        e = torch.exp(e - max(e, dim=-1, keepdims=True))
        a = e / torch.sum(e, dim=-1, keepdims=True)

        v = torch.bmm(a, inputs)

        return [v, a] if self.return_attention else v

    def _call_additive_emission(self, inputs):
        input_shape = list(inputs.shape)
        batch_size, input_len = input_shape[0], input_shape[1]

        q = expand_dims(torch.matmul(inputs, self.Wt), 2)
        k = expand_dims(torch.matmul(inputs, self.Wx), 1)
        if self.bias:
            h = torch.tanh(q + k + self.bh)
        else:
            h = torch.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.attention_bias:
            e = torch.reshape(torch.matmul(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = torch.reshape(torch.matmul(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs: Tensor):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        if self.attention_bias:
            e = torch.bmm(torch.matmul(inputs, self.Wa), inputs.swapaxes(1, 2))
        else:
            e = torch.bmm(torch.matmul(inputs, self.Wa), inputs.swapaxes(1, 2)) + self.ba
        return e
# end
