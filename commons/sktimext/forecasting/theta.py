import sktime.forecasting.theta as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class ThetaForecaster(sktf.ThetaForecaster):

    def __init__(
        self,
        prediction_length,
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
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(values=list(range(prediction_length)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
