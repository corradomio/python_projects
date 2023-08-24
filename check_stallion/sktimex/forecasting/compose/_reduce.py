from sktime.forecasting.compose import *
from sktime.forecasting.compose._reduce import _check_strategy, _check_scitype, _infer_scitype
from sktime.utils.estimators.dispatch import construct_dispatch
from ._trf import TabularRegressorForecaster


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _get_forecaster(scitype, strategy):
    registry = {
        "tabular-regressor": {
            "multioutput": MultioutputTabularRegressionForecaster,
            "direct": TabularRegressorForecaster,  # <<<
            "recursive": TabularRegressorForecaster,  # <<<
            "dirrec": TabularRegressorForecaster,  # <<<
            # "direct": DirRecTabularRegressionForecaster,
            # "recursive": RecursiveTabularRegressionForecaster,
            # "dirrec": DirRecTabularRegressionForecaster,
        },
        "time-series-regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
            "multioutput": MultioutputTimeSeriesRegressionForecaster,
            "dirrec": DirRecTimeSeriesRegressionForecaster,
        },
    }
    return registry[scitype][strategy]


# ---------------------------------------------------------------------------
# make_reduction
# ---------------------------------------------------------------------------

def make_reduction(
    estimator,
    strategy="recursive",
    window_length=10,
    scitype="infer",
    transformers=None,
    pooling="local",
    windows_identical=True
):
    # We provide this function as a factory method for user convenience.
    strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)

    dispatch_params = {
        "estimator": estimator,
        "window_length": window_length,
        "transformers": transformers,
        "pooling": pooling,
        "windows_identical": windows_identical,
    }

    return construct_dispatch(Forecaster, dispatch_params)
