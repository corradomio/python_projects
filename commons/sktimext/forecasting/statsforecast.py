import sktime.forecasting.statsforecast as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative


class StatsForecastAutoARIMA(sktf.StatsForecastAutoARIMA):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class StatsForecastAutoCES(sktf.StatsForecastAutoCES):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class StatsForecastAutoETS(sktf.StatsForecastAutoETS):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class StatsForecastAutoTBATS(sktf.StatsForecastAutoTBATS):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class StatsForecastAutoTheta(sktf.StatsForecastAutoTheta):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class StatsForecastMSTL(sktf.StatsForecastMSTL):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class StatsForecastADIDA(sktf.StatsForecastADIDA):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
