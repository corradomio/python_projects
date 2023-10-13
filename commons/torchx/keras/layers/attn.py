from ... import nn as nnx


class SeqSelfAttention(nnx.SeqSelfAttention):

    def __init__(self, input, units, **kwargs):
        super().__init__(in_features=input, out_features=units, **kwargs)