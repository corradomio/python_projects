import math
import torch
from torch import nn
from d2l import torch as d2l

# (batch, seq, data)    same as     RNN

num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
X = torch.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens), (batch_size, num_queries, num_hiddens))
