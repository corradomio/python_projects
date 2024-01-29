from typing import Union

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def dim_of(dim: Union[int, list[int]]):
    return [dim] if isinstance(dim, int) else dim


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def is_shape(s):
    """Check if s is a 2-elements tuple """
    return isinstance(s, (list, tuple)) and (len(s) == 2)


def assert_is_tuple(v, name, len=None):
    """Check if v is a tuple with the specified number of elements"""
    assert isinstance(v, (list, tuple)), f"The parameter {name} must be a tuple list: {v}"
    if len is not None:
        assert len(v) == len, f"The parameter {name} must be a tuple of length {len}: {v}"


# ---------------------------------------------------------------------------
# torch Tensor Utilities
# ---------------------------------------------------------------------------

def to_tensor(v: np.ndarray, dtype=torch.float32, device=None) -> Tensor:
    if len(v.shape) == 1:
        v = v.reshape((-1, 1))
    t = torch.from_numpy(v)
    if dtype is not None:
        t = t.type(dtype)
    if device is not None:
        t = t.to(device)
    return t


def time_repeat(x: Tensor, n: int) -> Union[Tensor, list[Tensor]]:
    """
    Repeat the tensor x along the time axis (dim=1), as used in RNN layers
    :param x: tensor to replicate
    :param n: n of replicas
    :return: replicated tensor with shape (t.shape[0], n, t.shape[1])
    """
    if isinstance(x, tuple):
        return [time_repeat(t, n) for t in x]
    batch, data_size = x.shape
    r = x.repeat((1, 1, n))
    r = r.reshape((batch, n, data_size))
    return r


def expand_dims(t: Tensor, dim: Union[list[int], int]=-1) -> Tensor:
    if isinstance(dim, int):
        return torch.unsqueeze(t, dim)
    for d in dim:
        t = torch.unsqueeze(t, d)
    return t


def remove_dims(t: Tensor, dim: Union[list[int], int]=-1) -> Tensor:
    if isinstance(dim, int):
        return torch.squeeze(t, dim)
    for d in dim:
        t = torch.squeeze(t, d)
    return t


def cast(t: Tensor, dtype) -> Tensor:
    return t.to(dtype)


def max(t: Tensor, dim=None, keepdims=False):
    """
    Return the maximum value in the tensor, discarges the index information
    :param t: tensor
    :param dim: axes where to compute the value
    :param keepdims: if to keep all dimensions
    :return: the maximum value
    """
    max_vals, max_idx = torch.max(t, dim=dim, keepdims=keepdims)
    return max_vals


def split(t: Tensor, splits: list[int], dim=1):
    """
    Equivalent to 'tensorflow.split(t, [d0, d1, ...])'
    :param t: tensor to split
    :param splits: width of each dimension
    :return: list of sub-tensors
    """
    assert dim == 1, "It is supported only 'dim=1'"

    if isinstance(splits, int):
        n = t.shape[dim] // splits
        splits = [splits] * n

    parts = []
    i = 0
    for s in splits:
        p = t[:, i:i+s]
        parts.append(p)
        i += s
    return parts


def norm(data, p=1) -> torch.Tensor:
    t = torch.tensor(data)
    if torch.abs(t).sum() == 0:
        return t
    if p == 1:
        t = t / t.sum()
    elif p == 2:
        t = t / torch.sqrt(torch.dot(t, t).sum())
    else:
        t = t / torch.pow(torch.pow(t, p).sum(), 1./p)

    return t


# ---------------------------------------------------------------------------
# print_shape
# ---------------------------------------------------------------------------

PRINT_SHAPE = set()


def _print_shape(what, x, i=0):
    if isinstance(x, (list, tuple)):
        if i == 0:
            print("  "*i, what, "...")
        else:
            print("  " * i, "...")
        for t in x:
            _print_shape(what, t, i+1)
        return
    if i == 0:
        print("  "*i, what, tuple(x.shape))
    else:
        print("  " * i, tuple(x.shape))


def print_shape(x, what=None):
    global PRINT_SHAPE
    if x is None:
        PRINT_SHAPE = set()
        return
    elif what in PRINT_SHAPE:
        return

    PRINT_SHAPE.add(what)
    _print_shape(what, x)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
