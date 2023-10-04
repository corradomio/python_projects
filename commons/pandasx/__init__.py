from .base import *
from .time import infer_freq, set_datetime_index
from .io import read_data
from .missing import nan_replace

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Add 'DatetimeIndex.week' compatibility property
#

def _dtix_week(self: pd.DatetimeIndex):
    return pd.Index(self.isocalendar().week.to_numpy(dtype=np.int32))


if not hasattr(pd.DatetimeIndex, 'week'):
    pd.DatetimeIndex.week = property(fget=_dtix_week)

