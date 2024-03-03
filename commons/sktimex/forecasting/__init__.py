from .linear import *
from .scikit import *

try:
    # available only if 'torch/skorch are available
    from .nn import compute_input_output_shapes
    from .cnn import CNNLinearForecaster
    from .rnn import RNNLinearForecaster
    from .lnn import LinearNNForecaster
    from .skorch import SkorchForecaster
except:
    pass
