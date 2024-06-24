
__all__ = [
    'Seq2SeqNetwork'
]

import torch.nn as nn

from stdlib import kwselect
from torchx.nn.modules.rnn import RNNX_FLAVOURS, RNNX_PARAMS
from torchx.utils import time_repeat


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _create_rnn(flavour, all_params, prefix):
    rnn_class = RNNX_FLAVOURS[flavour]
    rnn_params = kwselect(all_params, RNNX_PARAMS)
    rnn = rnn_class(**rnn_params)
    return rnn


def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.GRU) or isinstance(module, nn.RNN) or isinstance(module, nn.LSTM):
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])


def assert_seq_size(dim, name):
    assert isinstance(dim, (list, tuple)) and len(dim) == 2, f"Parameter {name} must be (seq_len, data_size)"


# ---------------------------------------------------------------------------
# Seq2SeqNet
# ---------------------------------------------------------------------------
#
#   x -> [encoder] -> status -> [decoder] -> y
#
#   (batch, input_seq, input_size)    -> (batch, output_seq, output_size)
#

class Seq2SeqNetwork(nn.Module):

    def __init__(self,
                 input_size,        # must be   (seq, input_size)
                 hidden_size=None,
                 output_seq=None,
                 flavour="gru",
                 **kwargs
                 ):
        super().__init__()
        self.output_seq = output_seq

        if hidden_size is None:
            hidden_size = input_size

        # it must return the state and just the last prediction
        self.encoder = _create_rnn(flavour, kwargs | {'input_size': input_size, 'hidden_size': hidden_size,
                                                      'return_state': True, 'return_sequence': False},
                                   "encoder__")
        self.decoder = _create_rnn(flavour, kwargs | {'input_size': hidden_size, 'hidden_size': hidden_size},
                                   "decoder__")

        # initialize the layers parameters
        self.apply(init_seq2seq)

    def forward(self, input):
        context1, state1 = self.encoder(input)
        if self.output_seq:
            output_seq = self.output_seq
        elif self.encoder.batch_first:
            output_seq = input.shape[1]
        else:
            output_seq = input.shape[0]

        if self.encoder.batch_first:
            batch_len = input.shape[0]
        else:
            batch_len = input.shape[1]

        # the last prediction & state of the encoder is replicated
        # the number of times equals to output_seq

        contextn = time_repeat(context1, output_seq)
        staten = time_repeat(state1, batch_len)
        output = self.decoder(contextn, staten)
        return output

