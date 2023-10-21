#
# d2l, Chapter 11
#


import torch
import torch.nn as nn
import math
from d2l import torch as d2l


#
# Visualization
#

#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
    cmap='Reds'):
    """Show heatmaps of matrices."""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
                if titles:
                    ax.set_title(titles[j])
                    fig.colorbar(pcm, ax=axes, shrink=0.6);


attention_weights = torch.eye(10).reshape((1, 1, 10, 10))

# show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
# d2l.plt.show()


#
# Attention Pooling by Similarity
#
d2l.use_svg_display()


# Define some kernels
def gaussian(x):
    # exp(-(x^2)/2)
    return torch.exp(-x**2 / 2)


def boxcar(x):
    # |x| < 1
    return torch.abs(x) < 1.0


def constant(x):
    # 1 + 0*x = 0
    # trick to have a value differentiable
    return 1.0 + 0 * x


def epanechikov(x):
    # max(0, 1-|x|)
    return torch.max(1 - torch.abs(x), torch.zeros_like(x))


#%%

kernels = (gaussian, boxcar, constant, epanechikov)
names = ('Gaussian', 'Boxcar', 'Constant', 'Epanechikov')

# fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
# x = torch.arange(-2.5, 2.5, 0.1)
# for kernel, name, ax in zip(kernels, names, axes):
#     ax.plot(x.detach().numpy(), kernel(x).detach().numpy())
#     ax.set_xlabel(name)
# d2l.plt.show()


#%%

def f(x):
    return 2 * torch.sin(x) + x


n = 40
x_train, _ = torch.sort(torch.rand(n) * 5)
y_train = f(x_train) + torch.randn(n)
x_val = torch.arange(0, 5, 0.1)
y_val = f(x_val)

# d2l.plt.scatter(x_train, y_train)
# d2l.plt.scatter(x_val, y_val)
# d2l.plt.show()


#
# Attention Pooling via Nadaraya–Watson Regression
#
#   [1 3]     [6 8]    [1*6 + 2*4 = 14 | 1*8 + 3*2 = 14]
#   [7 5]     [4 2]    [7*6 + 5*4 = 62 | 7*8 + 5*2 = 66]
#

def nadaraya_watson(x_train, y_train, x_val, kernel):
    dists = x_train.reshape((-1, 1)) - x_val.reshape((1, -1))
    # Each column/row corresponds to each query/key
    k = kernel(dists).type(torch.float32)
    # Normalization over keys for each query
    attention_w = k / k.sum(0)

    # a @  b    == __matmul__
    # a @= b    == __imatmul__
    y_hat = y_train@attention_w
    return y_hat, attention_w


def plot(x_train, y_train, x_val, y_val, kernels, names, attention=False):
    fig, axes = d2l.plt.subplots(1, 4, sharey=True, figsize=(12, 3))
    pcm = None
    for kernel, name, ax in zip(kernels, names, axes):
        y_hat, attention_w = nadaraya_watson(x_train, y_train, x_val, kernel)
        if attention:
            pcm = ax.imshow(attention_w.detach().numpy(), cmap='Reds')
        else:
            ax.plot(x_val, y_hat)
            ax.plot(x_val, y_val, 'm--')
            ax.plot(x_train, y_train, 'o', alpha=0.5);
        ax.set_xlabel(name)
        if not attention:
            ax.legend(['y_hat', 'y'])
    if attention:
        fig.colorbar(pcm, ax=axes, shrink=0.7)


# plot(x_train, y_train, x_val, y_val, kernels, names)
# d2l.plt.tight_layout()
# d2l.plt.show()


# plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
# d2l.plt.tight_layout()
# d2l.plt.show()


#
# Adapting Attention Pooling
#

sigmas = (0.1, 0.2, 0.5, 1)
names = ['Sigma ' + str(sigma) for sigma in sigmas]


def gaussian_with_width(sigma):
    return (lambda x: torch.exp(-x**2 / (2*sigma**2)))


kernels = [gaussian_with_width(sigma) for sigma in sigmas]

# plot(x_train, y_train, x_val, y_val, kernels, names)
# d2l.plt.show()


# plot(x_train, y_train, x_val, y_val, kernels, names, attention=True)
# d2l.plt.show()


#
# Attention Scoring Functions
#

# Masked Softmax Operation

def masked_softmax(X, valid_lens): #@save
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0.):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))

print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))


# Batch Matrix Multiplication

Q = torch.ones((2, 3, 4))
K = torch.ones((2, 4, 6))
d2l.check_shape(torch.bmm(Q, K), (2, 3, 6))


#
# Scaled Dot Product Attention
#

class DotProductAttention(nn.Module): #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


queries = torch.normal(0, 1, (2, 1, 2))
keys = torch.normal(0, 1, (2, 10, 2))
values = torch.normal(0, 1, (2, 10, 4))
valid_lens = torch.tensor([2, 6])

attention = DotProductAttention(dropout=0.5)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))


# d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
# d2l.plt.show()


#
# Additive Attention
#
class AdditiveAttention(nn.Module): #@save
    """Additive attention."""
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)


queries = torch.normal(0, 1, (2, 1, 20))
attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
attention.eval()
d2l.check_shape(attention(queries, keys, values, valid_lens), (2, 1, 4))


d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)), xlabel='Keys', ylabel='Queries')
d2l.plt.show()
