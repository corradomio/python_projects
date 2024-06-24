import sktime.forecasting.compose as sfcr
from sktime.forecasting.compose import RecursiveTimeSeriesRegressionForecaster
from sktime.forecasting.compose._reduce import _check_strategy, _check_scitype, _infer_scitype
from sktime.forecasting.compose._reduce import _DirectReducer, _MultioutputReducer, _DirRecReducer
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.estimators.dispatch import construct_dispatch
from ._trf import (FlexibeDirectRegressionForecaster, FlexibleMultioutputRegressionForecaster,
                   FlexibleRecursiveRegressionForecaster, FlexibleDirRecRegressionForecaster)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class RecursiveTabularRegressionForecaster(sfcr.RecursiveTabularRegressionForecaster):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "tabular-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length + 1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


class DirectTabularRegressionForecaster(sfcr.DirectTabularRegressionForecaster):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "tabular-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
        windows_identical=True,
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
            windows_identical=windows_identical
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length+1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


class DirRecTabularRegressionForecaster(sfcr.DirRecTabularRegressionForecaster):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "tabular-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length + 1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


class MultioutputTabularRegressionForecaster(sfcr.MultioutputTabularRegressionForecaster):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "tabular-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length + 1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class DirectTimeSeriesRegressionForecaster(_DirectReducer):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "time-series-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
        windows_identical=True,
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
            windows_identical=windows_identical
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length + 1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


class MultioutputTimeSeriesRegressionForecaster(_MultioutputReducer):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "time-series-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length + 1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


class DirRecTimeSeriesRegressionForecaster(_DirRecReducer):
    _tags = {
        "requires-fh-in-fit": False,  # is the forecasting horizon required in fit?
    }
    _estimator_scitype = "time-series-regressor"

    def __init__(
        self,
        estimator,
        window_length=10,
        prediction_length=1,
        transformers=None,
        pooling="local",
    ):
        super().__init__(
            estimator=estimator,
            window_length=window_length,
            transformers=transformers,
            pooling=pooling,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(list(range(1, prediction_length + 1)))

    def _fit(self, y, X=None, fh=None):
        return super()._fit(y=y, X=X, fh=fh if fh is not None else self._fh_in_fit)


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _get_forecaster(scitype, strategy):
    registry = {
        "tabular-regressor": {
            # "direct": DirectTabularRegressionForecaster,
            # "recursive": RecursiveTabularRegressionForecaster,
            # "multioutput": MultioutputTabularRegressionForecaster,
            # "dirrec": DirRecTabularRegressionForecaster,

            "direct": FlexibeDirectRegressionForecaster,
            "recursive": FlexibleRecursiveRegressionForecaster,
            "multioutput": FlexibleMultioutputRegressionForecaster,
            "dirrec": FlexibleDirRecRegressionForecaster,
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
    prediction_length=1,
    scitype="infer",
    transformers=None,
    pooling="local",
    windows_identical=True
):
    # We provide this function as a factory method for user convenience.
    if strategy != "tabular":
        strategy = _check_strategy(strategy)
    scitype = _check_scitype(scitype)

    if scitype == "infer":
        scitype = _infer_scitype(estimator)

    Forecaster = _get_forecaster(scitype, strategy)

    dispatch_params = {
        "estimator": estimator,
        "window_length": window_length,
        "prediction_length": prediction_length,
        "transformers": transformers,
        "pooling": pooling,
        "windows_identical": windows_identical,
    }

    return construct_dispatch(Forecaster, dispatch_params)
