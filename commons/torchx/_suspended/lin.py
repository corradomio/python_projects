import torch
import torch.nn as nn
from torchx import nn as nnx


def assert_shape_2d(shape):
    assert isinstance(shape, tuple) and len(shape) == 2


# ---------------------------------------------------------------------------
# Seq2Seq: there are 2 NN:
#
#       encoder(Xt) -> ye, ct
#       decoder(ct) -> ye, yp
#
# Shapes
#
#       input_shape   (seq_len, |Xt|)       batch_first=True
#       output_shape  (seq_len, |y|)        batch_first=True
#
# if the TS has input features, |Xt| > |y| else |Xt| = |y|
# If |Xt| > |y|, we can have:
#
#       Xt = [X|y]      target_first=False
#       Xt = [y|X]      target_first=True
#
#  For fitting, it is possible to use:
#
#       model.fit(Xt, y)            model.predict(fh, Xt)
#
# teacher forcing
#
#       model.fit((Xt,y), y)        model.predict(fh, Xt)
#
#       model.fit((Xt,y), (yt,y))
#       model.fit((Xt,y), (Xt,y))   in this way the input is the same
#


# ---------------------------------------------------------------------------
# Seq2SeqLinear
# ---------------------------------------------------------------------------

class Seq2SeqLinear(nn.Module):

    def __init__(self, input_shape, output_shape, hidden_size=None, target_first=False):
        super().__init__()
        assert_shape_2d(input_shape)
        assert_shape_2d(output_shape)

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_size = hidden_size
        self.target_first = target_first

        if hidden_size is not None:
            self.enc = nnx.Linear(in_features=input_shape, out_features=hidden_size)
            self.dec = nnx.Linear(in_features=hidden_size, out_features=output_shape)
        else:
            self.lin = nnx.Linear(in_features=input_shape, out_features=output_shape)

    def forward(self, x):
        if self.hidden_size is None:
            return self.lin(x)
        else:
            t = self.enc(x)
            return self.dec(t)
# end


# ---------------------------------------------------------------------------
# Seq2SeqLinear
# ---------------------------------------------------------------------------

