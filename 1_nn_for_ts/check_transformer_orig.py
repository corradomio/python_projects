import logging.config
import os
import warnings

import torch as T
import matplotlib.pyplot as plt
import skorch

import pandasx as pdx
import sktimex
import torchx
import torchx.nn as nnx
from skorchx.callbacks.logging import PrintLog
from sktimex.utils.plotting import plot_series
from stdlib import lrange, lrange1
from torchx.nn.timeseries import *

DATA_DIR = "data"
DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
GROUPS = ['item_country']

N = 35  # batch size
Lin = 36  # input sequence length
Hin = 19  # n input features
Lout = 2  # output sequence length
Hout = 1  # n output features
Nh = 3  # n heades
Nb = 1  # n blocks


def main():
    # [35, 36, 19] -> [35, 2, 1]
    Xe = T.rand(N, Lin, Hin)

    Xd = T.rand(N, Lout, Hout)
    Yd = T.rand(N, Lout, Hout)

    pr = nnx.PositionalReplicate(Nh, Hin)

    # d_model: int = 512,
    # nhead: int = 8,
    # num_encoder_layers: int = 6,
    # num_decoder_layers: int = 6,
    # dim_feedforward: int = 2048,
    # dropout: float = 0.1,
    # activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
    # custom_encoder: Optional[Any] = None,
    # custom_decoder: Optional[Any] = None,
    # layer_norm_eps: float = 1e-5,
    # batch_first: bool = False,
    # norm_first: bool = False,
    # bias: bool = True,
    # device=None,
    # dtype=None

    Xenc = pr(Xe)
    Xdec = pr(Xd)
    Ydec = pr(Yd)

    t = nnx.Transformer(
        d_model=Nh*Hin,
        nhead=Nh,
        num_encoder_layers=Nb,
        num_decoder_layers=Nb,
        dim_feedforward=Hout
    )
    t.eval()
    Ypre = t(Xenc, Xdec)

    pass


if __name__ == "__main__":
    main()
