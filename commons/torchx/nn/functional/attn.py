#
# Multiple attention mechanisms
#
# https://lilianweng.github.io/posts/2018-06-24-attention/
#

from math import sqrt
from typing import Optional

from torch import Tensor, zeros, transpose, reshape, softmax, norm, einsum, div
from torch.nn.functional import tanh


# def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
#     # Efficient implementation equivalent to the following:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#     attn_bias = zeros(L, S, dtype=query.dtype)
#     if is_causal:
#         assert attn_mask is None
#         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#         attn_bias.to(query.dtype)
#
#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
#         else:
#             attn_bias += attn_mask
#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value

#
# Q: N,L,k
# K: N,S,k
# V: N,S,v
#

def t(x):
    return transpose(x, -2, -1)


#
# @: matrix      multiplication
# *: elementwise multiplication
#

#   softmax(W) @ V
def _weighted_value(attn_weight: Tensor, value: Tensor) -> Tensor:
    attn_weight = softmax(attn_weight, dim=-1)
    return attn_weight @ value


#   softmax(1/sqrt(dk) * Q @ K^T) @ V
#   1/sqrt(dk) * Q @ K^T
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, scale=None) -> Tensor:
    scale = 1 / sqrt(query.size(-1)) if scale is None else scale
    if scale == 1:
        attn_weight = query @ t(key)
    else:
        attn_weight = query @ t(key) * scale
    return _weighted_value(attn_weight, value)


#   softmax(Q @ K^T) @ V
#   Q @ K^T
def dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    attn_weight = query @ t(key)
    return _weighted_value(attn_weight, value)


#   softmax(Q @ W @ K^T) @ V
#   Q @ W @ K^T
def general_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, W: Tensor) -> Tensor:
    attn_weight = query @ W @ t(key)
    return _weighted_value(attn_weight, value)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Q: N,L,k
# K: N,S,k
#
# R = cat(Q, K): N,L,S,2k
#

def cat(Q, K):
    N, L, k = Q.shape
    Nk, S, kk = K.shape
    assert N == Nk and k == kk

    R = zeros(N, L * S, k + k)

    s = 0
    for i in range(L):
        R[:, s:s + S, :k] = Q[:, i:i + 1, :]
        R[:, s:s + S, k:] = K[:, :, :]
        s += S
    return t(R)

# end


def linear_cat_attention(query: Tensor, key: Tensor, value: Tensor,
                         v: Tensor, W: Tensor, bias: Optional[Tensor]=None) -> Tensor:
    B, L, k = query.shape
    S = key.shape[1]

    C = cat(query, key)
    attn_weight = W @ C
    attn_weight = v @ tanh(attn_weight)
    attn_weight = reshape(attn_weight, (B, L, S))
    if bias:
        attn_weight += bias
    return _weighted_value(attn_weight, value)


# ---------------------------------------------------------------------------

def cosine_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    qk = query @ t(key)
    nq = norm(query, dim=-1)
    nk = norm(query, dim=-1)
    nqk =  einsum("bi,bj->bij",nq, nk)
    attn_weight = div(qk, nqk)
    return _weighted_value(attn_weight, value)
