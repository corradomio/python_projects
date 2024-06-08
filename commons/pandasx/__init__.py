from .base import *
from .io import read_data, write_data, save, load
from .cat import unique_values
from .missing import nan_replace
from .preprocessing import *
from .periodic import periodic_encode, set_datetime_index, last_week_in_month
from .freq import infer_freq, FREQUENCIES
from .onehot import onehot_encode
from .binhot import binhot_encode
from .resample import resample
# register 'is_instance' extensions
from . import is_instance

import typing
import pandas as pd
import numpy as np
from stdlib import method_of

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


# ---------------------------------------------------------------------------
# __class_getitem__

if not hasattr(pd.Series, "__class_getitem__"):
    @classmethod
    def series_class_getitem(cls, item):
        return typing._GenericAlias(pd.Series, item)
    pd.Series.__class_getitem__ = series_class_getitem


if not hasattr(pd.DataFrame, "__class_getitem__"):
    @classmethod
    def dataframe_class_getitem(cls, item):
        if not isinstance(item, tuple):
            item = (item,)
        return typing._GenericAlias(pd.DataFrame, item)
    pd.DataFrame.__class_getitem__ = dataframe_class_getitem
