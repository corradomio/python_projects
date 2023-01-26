import datetime
import numpy as np
import pandas as pd
from pandas.core.indexes.numeric import NumericIndex


def is_valid_index(index):
    if len(index) <= 1:
        return True
    elif isinstance(index, pd.RangeIndex):
        return True
    elif isinstance(index, NumericIndex):
        data = index.to_numpy()
        step = index[1] - index[0]
        if step <= 0: return False
        same = np.arange(start=index[0], stop=index[-1]+1, step=index[1] - index[0], dtype=data.dtype)
        return len(data) == len(same) and all(data == same)
    if isinstance(index, pd.PeriodIndex):
        return True
    elif isinstance(index, pd.DatetimeIndex):
        data = index.to_numpy()
        start = data[0]
        stop = data[-1]
        step = data[1]-data[0]
        if step <= 0: return False
        pass

ii = pd.Index(data=[1, 4, 2, 5], dtype=int)
ij = pd.Index(data=[1, 3, 5, 7], dtype=int)
ri = pd.RangeIndex(start=0, stop=100, step=2)

print(is_valid_index(ii))
print(is_valid_index(ij))
print(is_valid_index(ri))

dti = pd.to_datetime(["1/1/2018", np.datetime64("2018-01-01"), datetime.datetime(2018, 1, 1)])
dri = pd.date_range("2018-01-01", periods=3, freq="H")
pi = pd.PeriodIndex(year=[2000, 2002], quarter=[1, 3])
pi = pd.period_range(start='2000-01-01', end='2023-12-31', freq='D')

dtime = dti[0]
period = pi[0]

is_valid_index(dti)
is_valid_index(dri)
is_valid_index(pi)

print("done")
