import sktime.forecasting.mapa as sktf
from sktime.forecasting.base import ForecastingHorizon
# from .fix_fh import fix_fh_relative
from .recpred import RecursivePredict

#
# fh_in_fit
#

class MAPAForecaster(sktf.MAPAForecaster, RecursivePredict):

    def __init__(
            self,
            sp=6,
            pred_len=1,
            aggregation_levels=None,
            base_forecaster=None,
            agg_method="mean",
            decompose_type="multiplicative",
            forecast_combine="mean",
            imputation_method="ffill",
            weights=None,
    ):
        super().__init__(
            sp=sp,
            aggregation_levels=aggregation_levels,
            base_forecaster=base_forecaster,
            agg_method=agg_method,
            decompose_type=decompose_type,
            forecast_combine=forecast_combine,
            imputation_method=imputation_method,
            weights=weights
        )
        self.pred_len = pred_len
        self._fh_in_fit = ForecastingHorizon(values=list(range(1, pred_len + 1)))

    def fit(self, y, X=None, fh=None):
        return super().fit(y, X=X, fh=self._fh_in_fit)

    def predict(self, fh=None, X=None):
        # fh = fix_fh_relative(fh)
        return self.recursive_predict(fh, X)
