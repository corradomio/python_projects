import sktime.forecasting.arch as sktf
from sktime.forecasting.base import ForecastingHorizon
from ..fix_fh import fix_fh_relative


class ARCH(sktf.ARCH):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred

