from .base import *
from .io import read_data, write_data
from .cat import unique_values
from .missing import nan_replace
from .preprocessing import *
from .periodicx import periodic_encode, set_datetime_index
from .freq import infer_freq, FREQUENCIES
from .onehot import onehot_encode
from .binhot import binhot_encode

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Add 'DatetimeIndex.week' compatibility property
#

def _dtix_week(self: pd.DatetimeIndex):
    return pd.Index(self.isocalendar().week.to_numpy(dtype=np.int32))


if not hasattr(pd.DatetimeIndex, 'week'):
    pd.DatetimeIndex.week = property(fget=_dtix_week)

