import sktime.forecasting.darts as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative


class DartsRegressionModel(sktf.DartsRegressionModel):

    def _create_forecaster(self: "DartsRegressionModel"):
        if self._X is None and self.lags_future_covariates is not None:
            self.lags_future_covariates = None
        return super()._create_forecaster()

    def fit(self, y, X=None, fh=None):
        if X is None:
            self.lags_future_covariates = None
        return super().fit(y, X, fh)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class DartsXGBModel(sktf.DartsXGBModel):

    def _create_forecaster(self: "DartsRegressionModel"):
        if self._X is None and self.lags_future_covariates is not None:
            self.lags_future_covariates = None
        return super()._create_forecaster()

    def fit(self, y, X=None, fh=None):
        if X is None:
            self.lags_future_covariates = None
        return super().fit(y, X, fh)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class DartsLinearRegressionModel(sktf.DartsLinearRegressionModel):

    def _create_forecaster(self: "DartsRegressionModel"):
        if self._X is None and self.lags_future_covariates is not None:
            self.lags_future_covariates = None
        return super()._create_forecaster()

    def fit(self, y, X=None, fh=None):
        if X is None:
            self.lags_future_covariates = None
        return super().fit(y, X, fh)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
