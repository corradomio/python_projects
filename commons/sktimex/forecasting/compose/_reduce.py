import pandas as pd
import sktime.forecasting.compose
from sktime.forecasting.compose import *
from sktime.forecasting.compose._reduce import _check_strategy, _check_scitype, _infer_scitype
from sktime.utils.estimators.dispatch import construct_dispatch


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class DirRecTabularRegressionForecasterExt(sktime.forecasting.compose.DirRecTabularRegressionForecaster):

    def _check_fh(self, fh):
        if not self._is_fitted:
            return super()._check_fh(fh)
        else:
            from sktime.utils.validation.forecasting import check_fh
            fh = check_fh(fh=fh, freq=self._cutoff)
            return fh

    def _predict(self, fh, X=None):
        if len(fh) <= len(self._fh):
            return super()._predict(fh=fh, X=X)

        assert fh.is_relative
        assert self._fh.is_relative

        lfh = len(fh)
        sts = len(self._fh)  # n of time slots in each step
        nts = lfh  # n of time slots to generate
        yt_list = []
        for t in range(0, nts, sts):
            u = min(t + sts, lfh)
            Xt = X.iloc[t:u] if X is not None else None
            yt = super()._predict(fh=self._fh, X=Xt)
            super().update(y=yt, X=Xt, update_params=False)
            yt_list.append(yt)
        # end
        yt = pd.concat(yt_list)
        return yt


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

def _get_forecaster(scitype, strategy):
    registry = {
        "tabular-regressor": {
            "direct": DirRecTabularRegressionForecasterExt,             # <<<
            "recursive": RecursiveTabularRegressionForecaster,
            "multioutput": MultioutputTabularRegressionForecaster,
            "dirrec": DirRecTabularRegressionForecasterExt,             # <<<
        },
        "time-series-regressor": {
            "direct": DirectTimeSeriesRegressionForecaster,
            "recursive": RecursiveTimeSeriesRegressionForecaster,
            "multioutput": MultioutputTimeSeriesRegressionForecaster,
            "dirrec": DirRecTimeSeriesRegressionForecaster,
        },
    }
    return registry[scitype][strategy]


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

