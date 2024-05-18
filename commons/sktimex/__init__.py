import sktime.forecasting.base
from .utils import method_of, clear_yX
from .transform import *
from .lags import resolve_lags, resolve_tlags, LagSlots


# ---------------------------------------------------------------------------
# Add 'BaseForecaster.predict_history' method
#

@method_of(sktime.forecasting.base.BaseForecaster)
def predict_history(self, fh, X=None, yh=None, Xh=None, update_params=False):
    if yh is not None:
        self.update(y=yh, X=Xh, update_params=update_params)
    return self.predict(fh=fh, X=X)
# end

