# from .utils import method_of, clear_yX
# from .transform.lagt import LagsTrainTransform, LagsPredictTransform
# from .transform.lint import LinearTrainTransform, LinearPredictTransform

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

from sktime.forecasting.base import ForecastingHorizon, BaseForecaster
import pandas as pd
import numpy as np


#
# If fh._values is a pandas Index/RangeIndex, sometimes fh is
# considered "relative", but this is an ERROR!!!
#
def fix_fh_relative(fh: ForecastingHorizon) -> ForecastingHorizon:
    if not isinstance(fh, ForecastingHorizon):
        fh = ForecastingHorizon(fh)
    if isinstance(fh._values, pd.Index) and fh.is_relative and isinstance(fh[0], (int, np.int64)) and fh[0] != 1:
        fh._is_relative = False
    return fh

def predict_fix_fh(self, fh: ForecastingHorizon, X=None):
    fh_fixed = fix_fh_relative(fh)
    y_pred = self.predict_fix(fh_fixed, X)
    return y_pred

BaseForecaster.predict_fix = BaseForecaster.predict
BaseForecaster.predict = predict_fix_fh
