from sktime.forecasting.base import BaseForecaster
from .utils import method_of, clear_yX
from .transform import *
# from .forecasting import *
from .lags import resolve_lags, resolve_tlags, LagSlots


# ---------------------------------------------------------------------------
# Add 'BaseForecaster.predict_history' method
#

@method_of(BaseForecaster)
def predict_history(self, fh, X=None, yh=None, Xh=None):
    if yh is not None:
        self.update(y=yh, X=Xh, update_params=False)
    return self.predict(fh=fh, X=X)
# end

