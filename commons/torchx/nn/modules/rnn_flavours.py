from .rnn import *

# ---------------------------------------------------------------------------
# create_rnn
# ---------------------------------------------------------------------------

RNNX_FLAVOURS = {
    'lstm': LSTM,
    'gru': GRU,
    'rnn': RNN,
}

RNNX_PARAMS = [
    'input_size', 'hidden_size', 'num_layers', 'bidirectional', 'bias', 'dropout',
    # extended parameters
    'return_sequence', 'return_state'
]


def create_rnn(flavour: str, **kwargs):
    if flavour not in RNNX_FLAVOURS:
        raise ValueError(f"Unsupported RNN flavour {flavour}")
    else:
        return RNNX_FLAVOURS[flavour](**kwargs)
# end
