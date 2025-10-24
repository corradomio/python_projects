import sktime.forecasting.ltsf as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class LTSFLinearForecaster(sktf.LTSFLinearForecaster):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class LTSFDLinearForecaster(sktf.LTSFDLinearForecaster):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class LTSFNLinearForecaster(sktf.LTSFNLinearForecaster):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred


class LTSFTransformerForecaster(sktf.LTSFTransformerForecaster):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
