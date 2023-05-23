import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
#
#     Args:
#         input_size: The number of expected features in the input `x`
#         hidden_size: The number of features in the hidden state `h`
#         num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
#             would mean stacking two LSTMs together to form a `stacked LSTM`,
#             with the second LSTM taking in outputs of the first LSTM and
#             computing the final results. Default: 1
#         bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
#             Default: ``True``
#         batch_first: If ``True``, then the input and output tensors are provided
#             as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
#             Note that this does not apply to hidden or cell states. See the
#             Inputs/Outputs sections below for details.  Default: ``False``
#         dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
#             LSTM layer except the last layer, with dropout probability equal to
#             :attr:`dropout`. Default: 0
#         bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
#         proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

class LSTM(nn.LSTM):

    def __init__(self, *, input_size, hidden_size, num_layers, **kwargs):
        super().__init__(input_size, hidden_size, num_layers, **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden = None

    def forward(self, input, hx=None):
        hidden_state = torch.zeros(self.num_layers, input.shape[0], self.hidden_size)
        cell_state = torch.zeros(self.num_layers, input.shape[0], self.hidden_size)
        self.hidden = (hidden_state, cell_state)

        input, h = super().forward(input, self.hidden)

        return input
# end
