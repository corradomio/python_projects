from torch import Tensor
import torchx.nn as nnx


class Seq2Seq(nnx.Module):
    def __init__(self,
                 input_size, hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.,
                 bidirectional=False,
                 batch_first=True,
                 return_sequence=True,
                 return_state=False,
                 nonlinearity="tanh",
                 **kwargs):
        super().__init__()
        self.encoder = nnx.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            return_sequence=return_sequence,
            return_state=return_state,
            nonlinearity=nonlinearity,
            **kwargs
        )
        self.decoder = nnx.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
            return_sequence=return_sequence,
            return_state=return_state,
            nonlinearity=nonlinearity,
            **kwargs
        )
    # end

    def forward(self, inputs_outputs: tuple[Tensor, Tensor]) -> Tensor:
        inputs, outputs = inputs_outputs

        enc, hx = self.encoder.forward(inputs)
        dec, _ = self.decoder.forward(enc, hx)
        return dec
    # end
# end
