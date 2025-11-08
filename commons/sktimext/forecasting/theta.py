import sktime.forecasting.theta as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative
from .recpred import RecursivePredict

#
# fh_in_fit
#

class ThetaForecaster(sktf.ThetaForecaster, RecursivePredict):

    def __init__(
        self,
        pred_len=1,
        initial_level=None,
        deseasonalize=True,
        sp=1,
        deseasonalize_model="multiplicative",
    ):
        super().__init__(
            initial_level=initial_level,
            deseasonalize=deseasonalize,
            sp=sp,
            deseasonalize_model=deseasonalize_model,
        )
        self.pred_len = pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len+1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)
