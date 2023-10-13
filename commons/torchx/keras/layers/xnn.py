from ... import nn as nnx


# ---------------------------------------------------------------------------
# Note: implementation moved in torchx.nn.xnn
#       definitions here for back compatibility
# ---------------------------------------------------------------------------

class Reshape(nnx.ReshapeVector):
    """
    Keras compatible ReshapeVector
    """
    def __init__(self, shape=None, n_dims=0):
        super().__init__(shape=shape, n_dims=n_dims)


class RepeatVector(nnx.RepeatVector):
    """
    Keras compatible RepeatVector
    """
    def __init__(self, n_repeat=1):
        super().__init__(n_repeat=n_repeat)


class TimeDistributed(nnx.TimeDistributed):
    """
    Keras compatible TimeDistributed
    """
    def __init__(self, *models):
        super().__init__(*models)


class ChannelDistributed(nnx.ChannelDistributed):
    """
    Keras compatible ChannelDistributed
    """
    def __init__(self, *models):
        super().__init__(*models)
