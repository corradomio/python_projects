import torch as T
from torch import Tensor
from torch.nn.functional import softmax
from torch.nn.functional import tanh


def t(x: Tensor) -> Tensor:
    return T.transpose(x, -2, -1)


# -----------------------------------

B = 4
L = 3
S = 5
k = 2
v = 7

# Q: B x L x k
# K: B x S x k
# V: B x S x v
# R: B x L x v

# -----------------------------------

Q = T.rand(B, L, k)     # 4,3,2
K = T.rand(B, S, k)     # 4,5,2
V = T.rand(B, S, v)     # 4,5,7
R = T.rand(B, L, v)     # 4,3,7

Rshape = [B, L, v]

# -----------------------------------

# R = softmax(Q @ t(K), dim=-1) @ V
# print([B, L, v], list(R.shape))

# -----------------------------------

# W = T.rand(k, k)
# R = softmax(Q @ W @ t(K), dim=-1) @ V
# print([B, L, v], list(R.shape))

# -----------------------------------
# -----------------------------------

# Q = T.rand(B, S, k)
# K = T.rand(B, S, k)
# V = T.rand(B, S, v)
#
#
# Wq = T.rand(k, S)
# Wk = T.rand(k, S)
# vv = T.rand(1, S)
#
# R = softmax(tanh(Q @ Wq + K @ Wk), dim=-1) @ V
# print(R.shape)


# -----------------------------------
#   S          S     = S*L
# +----+....+----+
# | Q1 |    | QL | k
# +----+... +----+
# | K  |    | K  | k
# +----+    +----+
#
#

def cat(Q, K):
    N, L, k = Q.shape
    Nk, S, kk = K.shape
    assert N == Nk and k == kk

    R = T.zeros(N, L*S, k+k)

    s = 0
    for i in range(L):
        R[:, s:s+S, :k] = Q[:, i:i+1, :]
        R[:, s:s+S, k:] = K[:, :, :]
        s += S
    return t(R)
# end


# C = cat(Q, K)
# print("C", C.shape)
#
# W = T.rand(v, 2*k)
# R = W @ C
# print("W.C", R.shape)
#
# vv = T.rand(v, 1)
#
# R = t(vv) @ tanh(W @ cat(Q, K))
# print("v^T tanh(W.C)", R.shape)
#
# R = T.reshape(R, (B, L, S))
# print("v^T tanh(W.C)", R.shape)
#
# R = softmax(R, dim=-1) @ V
# print("softmax(v^T tanh(W.C)) V", R.shape, [B, L, v])



# R = softmax(t(vv) @ tanh(cat(Q, K) @ W), dim=-1)
# print("softmax", R.shape)
#
# R = softmax(t(vv) @ tanh(cat(Q, K) @ W), dim=-1) @ V
# print("attn", R.shape, Rshape)


# -----------------------------------

QK = Q @ t(K)
print("Q.K^T", QK.shape)

nQ = T.norm(Q, dim=-1)
print("|Q|", nQ.shape)
nK = T.norm(K, dim=-1)
print("|K|", nK.shape)

nQK =  T.einsum("bi,bj->bij", nQ, nK)
print("nQ outer nK", nQK.shape)

R = T.div(QK, nQK)
print("R", R.shape)

