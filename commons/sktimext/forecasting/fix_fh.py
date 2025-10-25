from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
from pandas import Index


#
# If fh._values is a pandas Index/RangeIndex, sometimes fh is
# considered "relative", but this is an ERROR!!!
#
def fix_fh_relative(fh: ForecastingHorizon) -> ForecastingHorizon:
    if isinstance(fh._values, Index) and fh.is_relative:
        fh._is_relative = False
    return fh


def _predict_recursive(model: BaseForecaster, fh: ForecastingHorizon, Xf, fhp: ForecastingHorizon, yp, Xp):
    nfh = len(fh)
    fhp = fhp.to_relative(model.cutoff)

    y_pred = model._predict()
