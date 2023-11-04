from .linear import LinearForecaster
from .scikit import ScikitForecaster

try:
    # available only if 'torch/skorch are available
    from .cnn import SimpleCNNForecaster, MultiLagsCNNForecaster
    from .rnn import SimpleRNNForecaster, MultiLagsRNNForecaster
except:
    pass

#
# Extensions/compatibility
#
from .linear import LinearForecastRegressor
from .scikit import ScikitForecastRegressor
