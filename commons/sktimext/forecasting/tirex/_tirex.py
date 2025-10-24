import sktime.forecasting.tirex as sktf
from sktime.forecasting.base import ForecastingHorizon
from ..fix_fh import fix_fh_relative

#
# Dependencies: tirex, dacite
# Problems with the license!
#

class TiRexForecaster(sktf.TiRexForecaster):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
