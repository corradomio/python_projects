import sktime.forecasting.mapa as sktf
from sktime.forecasting.base import ForecastingHorizon
from .fix_fh import fix_fh_relative

#
# fh_in_fit
#

class MAPAForecaster(sktf.MAPAForecaster):

    def __init__(
        self,
        prediction_length,
        aggregation_levels=None,
        base_forecaster=None,
        agg_method="mean",
        decompose_type="multiplicative",
        forecast_combine="mean",
        imputation_method="ffill",
        sp=6,
        weights=None,
    ):
        super().__init__(
            aggregation_levels=aggregation_levels,
            base_forecaster=base_forecaster,
            agg_method=agg_method,
            decompose_type=decompose_type,
            forecast_combine=forecast_combine,
            imputation_method=imputation_method,
            sp=sp,
            weights=weights,
        )
        self.prediction_length = prediction_length
        self._fh_in_fit = ForecastingHorizon(values=list(range(prediction_length)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def _predict(self, fh: ForecastingHorizon, X):
        fh = fix_fh_relative(fh)
        y_pred = super()._predict(fh, X)
        return y_pred
