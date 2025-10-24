import sktime.forecasting.bats as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# Incompatibility:
# ModuleNotFoundError: BATS requires package 'numpy<2' to be present in the python environment, with versions as
# specified, but incompatible version numpy 2.2.6 was found. This version requirement is not one by sktime, but
# specific to the module, class or object with name BATS(sp=12)..
#
class BATS(sktf.BATS):

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
