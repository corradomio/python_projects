#
# Pandas reader
# Some simple data set loaders: from csv and .arff data files
#
import datetime
import math
import random
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FREQUENCIES
# ---------------------------------------------------------------------------

FREQUENCIES = {
    # second
    'S': 1,

    # minute
    'T': 60,
    'min': 60,

    # hour
    'H': 60 * 60,
    'BH': 60 * 60,

    # day
    'B': 60 * 60 * 24,
    'D': 60 * 60 * 24,
    'C': 60 * 60 * 24,

    # week
    'W': 60 * 60 * 24 * 7,
    'BW': 60 * 60 * 24 * 5,

    # 15 days/half month
    'SM': 60 * 60 * 24 * 15,
    'SMS': 60 * 60 * 24 * 15,

    # month
    'M': 60 * 60 * 24 * 30,
    'BM': 60 * 60 * 24 * 30,
    'CBM': 60 * 60 * 24 * 30,
    'MS': 60 * 60 * 24 * 30,
    'MBS': 60 * 60 * 24 * 30,
    'CMBS': 60 * 60 * 24 * 30,

    # quarter
    'Q': 60 * 60 * 24 * 91,
    'BQ': 60 * 60 * 24 * 91,
    'QS': 60 * 60 * 24 * 91,
    'BQS': 60 * 60 * 24 * 91,

    # year
    'A': 60 * 60 * 24 * 365,
    'Y': 60 * 60 * 24 * 365,
    'BA': 60 * 60 * 24 * 365,
    'BY': 60 * 60 * 24 * 365,
    'AS': 60 * 60 * 24 * 365,
    'AY': 60 * 60 * 24 * 365,
    'BAS': 60 * 60 * 24 * 365,
    'BYS': 60 * 60 * 24 * 365,

    # 'WOM': 1,
    # 'LWOM': 1,

    # 'L': 1, 'ms': 1,    # milliseconds
    # 'U': 1, 'us': 1,    # microseconds
    # 'N': 1              # nanoseconds
}


# ---------------------------------------------------------------------------
# Statistical infer_freq
# ---------------------------------------------------------------------------
# https://pandas.pydata.org/docs/user_guide/timeseries.html
#
# B         business day frequency
# C         custom business day frequency
# D         calendar day frequency
# W         weekly frequency
# WOM       the x-th day of the y-th week of each month
# LWOM      the x-th day of the last week of each month
# M         month end frequency
# MS        month start frequency
# BM        business month end frequency
# BMS       business month start frequency
# CBM       custom business month end frequency
# CBMS      custom business month start frequency
# SM        semi-month end frequency (15th and end of month)
# SMS       semi-month start frequency (1st and 15th)
# Q         quarter end frequency
# QS        quarter start frequency
# BQ        business quarter end frequency
# BQS       business quarter start frequency
# REQ       retail (aka 52-53 week) quarter
# A, Y      year end frequency
# AS, YS    year start frequency
# AS, BYS   year start frequency
# BA, BY    business year end frequency
# BAS, BYS  business year start frequency
# RE        retail (aka 52-53 week) year
# BH        business hour frequency
# H         hourly frequency
# T, min    minutely frequency
# S         secondly frequency
# L, ms     milliseconds
# U, us     microseconds
# N         nanoseconds
#

def infer_freq(index, steps=5, ntries=3) -> str:
    """
    Infer 'freq' checking randomly different positions of the index

    :param index: pandas' index to use
    :param steps: number of success results
    :param ntries: maximum number of retries if some check fails
    :return: the inferred frequency
    """
    if isinstance(index, pd.Period):
        return index.freqstr
    if isinstance(index, pd.PeriodIndex):
        return index.iloc[0].freqstr
    if isinstance(index, pd.Series):
        return infer_freq(index.iloc[0], steps, ntries)

    n = len(index) - steps
    freq = None
    itry = 0
    while itry < ntries:
        i = random.randrange(n)
        tfreq = pd.infer_freq(index[i:i + steps])
        if tfreq is None:
            itry += 1
        elif tfreq != freq:
            freq = tfreq
            itry = 0
        else:
            itry += 1
    # end

    return freq


# ---------------------------------------------------------------------------
# datetime_periodic
# ---------------------------------------------------------------------------
# add different periodic columns to df

def dataframe_periodic(df: pd.DataFrame, freq: str, columns: list[str], method: str = 'circle',
                       datetime: Optional[str] = None) -> pd.DataFrame:
    """
    Add some extra column to represent a periodic time

    :param df: dataframe to process
    :param freq: frequency
    :param columns: column names to use. The n of columns depends on the method
    :param method: method to use
    :param datetime: if to use a datetime column or the DatetimeIndex/PeriodIndex
    :return:
    """
    if datetime is not None:
        date_values = df[datetime]
    else:
        date_values = df.index.to_series()

    if method in ['cossin', 'sincos', 'circle']:
        return circular_periodic(df, freq, columns, date_values)
    else:
        raise ValueError(f'Unsupported method {method}')


def circular_periodic(df: pd.DataFrame, freq: str, columns: list[str], date_values):
    assert len(df) == len(date_values)
    assert isinstance(columns, list) and len(columns) == 2
    dfreq = infer_freq(date_values)

    sfact = FREQUENCIES[freq]

    values = date_values.map(pd.Period.to_timestamp).map(pd.Timestamp.timestamp).to_numpy()
    values = np.mod(values, sfact)
    cos_v = np.cos(values * 2 * np.pi / sfact)
    sin_v = np.sin(values * 2 * np.pi / sfact)

    df[columns[0]] = cos_v
    df[columns[1]] = sin_v

    return df
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
