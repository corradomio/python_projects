from .linear import LinearForecaster
from .scikit import ScikitForecaster

try:
    # available only if 'torch/skorch are available
    from .cnn import SimpleCNNForecaster, MultiLagsCNNForecaster
    from .rnn import SimpleRNNForecaster, MultiLagsRNNForecaster
    from .lnn import LinearNNForecaster
except:
    pass

#
# Compatibility
#
from .nn import compute_input_output_shapes
from .linear import LinearForecastRegressor
from .scikit import ScikitForecastRegressor
