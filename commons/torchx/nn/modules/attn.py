#
# https://lilianweng.github.io/posts/2018-06-24-attention/
#
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_, xavier_normal_

from ..functional.attn import *


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def forward(self, query:Tensor, key:Tensor, value: Tensor) -> Tensor:
        ...


# ---------------------------------------------------------------------------
# DotAttention
# ---------------------------------------------------------------------------
# Luong2015
# Effective Approaches to Attention-based Neural Machine Translation
#
#   aij = xi^T yj
#

class DotProductAttention(Attention):

    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return dot_product_attention(query, key, value)
# end


# ---------------------------------------------------------------------------
# ScaledDotProduct
# ---------------------------------------------------------------------------
# Vaswani2017
# Attention Is All You Need
#
#          xi^T yj
#   aij = ----------
#          sqrt(k)
#

class ScaledDotProductAttention(Attention):

    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(query, key, value, scale=self.scale)
# end


# ---------------------------------------------------------------------------
# GeneralDotProductAttention
# ---------------------------------------------------------------------------
# Luong2015
# Effective Approaches to Attention-based Neural Machine Translation
#
#   aij = xi^T W yj
#

class GeneralDotProductAttention(Attention):

    def __init__(self, kdim):
        super().__init__()
        self.kdim = kdim

        self.weight = Parameter(torch.empty((kdim, kdim)))

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert self.kdim == query.shape[2] == key.shape[2], "Invalid 'kdim' parameter or 'query', 'key' dimensions"
        return general_dot_product_attention(query, key, value, self.weight)
# end


# ---------------------------------------------------------------------------
# AdditiveAttention
# ---------------------------------------------------------------------------
# Bahdanau2015
# NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
#
#       A = v^T tanh(W [Q;K] + b?)
#

class AdditiveAttention(Attention):

    def __init__(self, kdim, emb_dim, bias=False):
        super().__init__()
        self.kdim = kdim
        self.emb_dim = emb_dim
        self.bias = bias

        self.vT = Parameter(torch.empty(emb_dim))
        self.weight = Parameter(torch.empty((emb_dim, 2*kdim)))

        if bias:
            self.bias_weight = Parameter(torch.empty(emb_dim))
        else:
            self.bias_weight = None

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.vT)
        xavier_uniform_(self.weight)
        if self.bias:
            xavier_normal_(self.bias_weight)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        assert self.kdim == query.shape[2] == key.shape[2], "Invalid 'kdim' parameter or 'query', 'key' dimensions"
        return linear_cat_attention(query, key, value, self.vT, self.weight)
# end


# ---------------------------------------------------------------------------
# CosineAttention
# ---------------------------------------------------------------------------
# Graves2014
# Neural Turing Machines
#
#       A = cosine(Q, K)
#

class CosineAttention(Attention):

    def __init__(self):
        super().__init__()

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return cosine_attention(query, key, value)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
