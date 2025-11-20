from typing import Union

import pandas as pd
import datetime as dt
import numpy as np
from pandas import DateOffset
from pandas._libs.missing import NAType
from pandas._typing import Scalar

#
# Series.diff
#
#     Compute the difference of two elements in a Series.
# DataFrame.diff
#
#     Compute the difference of two elements in a DataFrame.
# Series.shift
#
#     Shift the index by some number of periods.
# DataFrame.shift
#
#     Shift the index by some number of periods.
#


# ---------------------------------------------------------------------------
# fractional_change
# ---------------------------------------------------------------------------
#
#   R[i] = (S[i] - S[i-1])/S[i-1]
#
#   R[0]: NaN

def fractional_change(
    data: Union[pd.DataFrame, pd.Series],
    periods: int = 1,
    fill_method: None = None,
    freq: DateOffset | dt.timedelta | str | None = None,
    fill_value: Scalar | NAType | None = None,
) -> Union[pd.DataFrame, pd.Series]:
    if isinstance(data, pd.DataFrame):
        return _fchange_df(data, periods, fill_method, freq, fill_value)
    if isinstance(data, pd.Series):
        return _fchange_ser(data, periods, fill_method, freq)
    else:
        raise ValueError(f"Unsupported data of type {type(data)}")


def _fchange_df(
    df: pd.DataFrame,
    periods: int = 1,
    fill_method: None = None,
    freq: DateOffset | dt.timedelta | str | None = None,
    fill_value: Scalar | NAType | None = None,
) -> pd.DataFrame:
    return df.pct_change(periods=periods, fill_method=fill_method, freq=freq, fill_value=fill_value)


def _fchange_ser(
    ser: pd.Series,
    periods: int = 1,
    fill_method: None = None,
    freq: DateOffset | dt.timedelta | str | None = None,
) -> pd.Series:
    return ser.pct_change(periods=periods, fill_method=fill_method, freq=freq)


# ---------------------------------------------------------------------------
# fractional_update
# ---------------------------------------------------------------------------
# 2 possibilities:
#   data[0] is NaN  -> it is enough 'value'
#   data[0] i value -> tu use 'value' and 'index'
#
# value: if Series, single value,
#        id DataFrame, array of values

def fractional_update(data: Union[pd.DataFrame, pd.Series], value: Union[pd.DataFrame, pd.Series]):
    assert isinstance(data, pd.DataFrame) and isinstance(value, pd.DataFrame) \
        or isinstance(data, pd.Series) and isinstance(value, pd.Series)
    if isinstance(data, pd.DataFrame):
        return _fupdate_df(data, value)
    if isinstance(data, pd.Series):
        return _fupdate_ser(data, value)
    else:
        raise ValueError(f"Unsupported data of type {type(data)}")


def _fupdate_df(df: pd.DataFrame, start: pd.DataFrame):
    n = len(df)
    m = len(df.columns)
    period = len(start)
    svals = start.values

    if df.iloc[0:period, :].isna().any().any():
        fvals = np.zeros((n, m), dtype=svals.dtype)
        fvals[:] = df.values
        fvals[:period, :] = svals
        index = df.index
    else:
        n += period
        fvals = np.zeros((n, m), dtype=svals.dtype)
        fvals[:period,:] = svals
        fvals[period:,:] = df.values
        index = start.index.union(df.index)

    for i in range(period, n):
        fvals[i] = (fvals[i]+1)*(fvals[i-period])

    return pd.DataFrame(data=fvals, index=index, columns=df.columns)




def _fupdate_ser(ser: pd.Series, start: pd.Series):
    n = len(ser)
    period = len(start)
    svals = start.values

    if ser[0:period].isna().any():
        fvals = np.zeros(n, dtype=svals.dtype)
        fvals[:] = ser.values
        fvals[:period] = svals
        index = ser.index
    else:
        fvals = np.zeros(period+n, dtype=svals.dtype)
        fvals[:period] = svals
        fvals[period:] = ser.values
        index = start.index.union(ser.index)

    for i in range(period, n):
        fvals[i] = (fvals[i]+1)*(fvals[i-period])

    return pd.Series(data=fvals, index=index, name=ser.name)

