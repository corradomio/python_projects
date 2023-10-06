from ...utils import KerasLayerMixin
from ... import nn as nnx


# ---------------------------------------------------------------------------
# Note: implementation moved in torchx.nn.xnn
#       definitions here for back compatibility
# ---------------------------------------------------------------------------

class RepeatVector(nnx.RepeatVector, KerasLayerMixin):
    def __init__(self, n_repeat=1):
        super().__init__(n_repeat=n_repeat)


class TimeDistributed(nnx.TimeDistributed, KerasLayerMixin):
    def __init__(self, *models):
        super().__init__(*models)


class ChannelDistributed(nnx.ChannelDistributed, KerasLayerMixin):
    def __init__(self, *models):
        super().__init__(*models)
