import sktime.forecasting.conditional_invertible_neural_network as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# Dependencies: FrEIA
#

class CINNForecaster(sktf.CINNForecaster):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
