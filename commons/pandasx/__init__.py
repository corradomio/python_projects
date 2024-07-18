import numpy as np
import pandas as pd

from .base import *
from .preprocessing import *
from .binhot import binhot_encode
from .cat import unique_values
from .freq import infer_freq, FREQUENCIES
from .io import read_data, write_data, save, load
from .missing import nan_replace
from .onehot import onehot_encode
from .periodic import periodic_encode, set_datetime_index, last_week_in_month
# from .resample import resample
from .dt import to_datetime, date_range, to_period, to_date_type
from .sql import read_sql, read_sql_query
from .to_json import to_json
from .spike import is_spike
from .hsched import is_heteroschedastic
from .upd import update


# ---------------------------------------------------------------------------
# Add 'DatetimeIndex.week' compatibility property
# Add missing 'PeriodIndex.weekinmonth', 'Period.weekinmonth'
# Monday = 0


if not hasattr(pd.DatetimeIndex, 'week'):
    def _dtix_week(self: pd.DatetimeIndex):
        return pd.Index(self.isocalendar().week.to_numpy(dtype=np.int32))
    pd.DatetimeIndex.week = property(fget=_dtix_week)


if not hasattr(pd.PeriodIndex, "weekinmonth"):
    def _i_weekinmonth(self: pd.PeriodIndex):
        day = self[0].day
        dow = self[0].dayofweek
        dow = (dow-day+1) % 7
        return pd.Index((self.day - 1 + dow) // 7)
        # return pd.Index((self.day - 1) // 7)
    pd.PeriodIndex.weekinmonth = property(fget=_i_weekinmonth)
    pd.DatetimeIndex.weekinmonth = property(fget=_i_weekinmonth)

    def _p_weekinmonth(self: pd.Period):
        day = self.day
        dow = self.dayofweek
        return (day - 1 + dow)//7
        # return (self.day - 1) // 7
    pd.Period.weekinmonth = property(fget=_p_weekinmonth)
