from .cnn import *


# ---------------------------------------------------------------------------
# create_cnn
# ---------------------------------------------------------------------------

CNNX_FLAVOURS = {
    'cnn': Conv1d,
}

CNNX_PARAMS = [
    'in_channels', 'out_channels',
    'kernel_size', 'stride', 'dilation',
    'groups', 'bias',
    'padding', 'padding_mode',
    'device', 'dtype',
    # extended parameters
    'channels_last'
]


def create_cnn(flavour: str, **kwargs):
    if flavour not in CNNX_FLAVOURS:
        raise ValueError(f"Unsupported CNN flavour {flavour}")
    else:
        return CNNX_FLAVOURS[flavour](**kwargs)
# end
