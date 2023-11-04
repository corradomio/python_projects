#
# A LSTM-based seq2seq model for time series forecasting
# https://medium.com/@shouke.wei/a-lstm-based-seq2seq-model-for-time-series-forecasting-3730822301c5
#

import pandasx as pdx
import torchx as tx

import torch.nn as nn
import torch


# Abstract seq2seq model in Python (using Pytorch)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        context_vector = self.encoder(inputs)
        outputs = self.decoder(context_vector)
        return outputs


def main():
    tx.print_shape(None, )

    data = pdx.read_data(
        '../data/ercot_data.csv',
        ignore_unnamed=True
    )

    pass
# end


if __name__ == "__main__":
    main()
