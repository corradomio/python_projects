from sktime.forecasting.base import ForecastingHorizon
from pandas import Index


#
# If fh._values is an pandas Index/RangeIndex, sometimes fh is
# considered "relative", but this is an ERROR!!!
#
def fix_fh_relative(fh: ForecastingHorizon):
    if isinstance(fh._values, Index) and fh.is_relative:
        fh._is_relative = False
    return fh