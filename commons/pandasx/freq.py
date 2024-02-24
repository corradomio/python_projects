import pandas as pd
import numpy as np
import random

# ---------------------------------------------------------------------------
# FREQUENCIES
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

FREQUENCIES = {
    # second
    'S': 1,                     # one second
    's': 1,                     # one second

    # minute
    'T': 60,
    'min': 60,                  # one minute

    # hour
    'H': 60 * 60,               # one hour
    'h': 60 * 60,               # one hour
    'BH': 60 * 60,              # business hour
    'CBH': 60 * 60,             # custom business hour

    # day
    'B': 60 * 60 * 24,          # business day (weekday)
    'D': 60 * 60 * 24,          # one absolute day
    'C': 60 * 60 * 24,          # custom business day

    # week
    'W': 60 * 60 * 24 * 7,      # one week, optionally anchored on a day of the week
    'BW': 60 * 60 * 24 * 5,
    'WOM': 60 * 60 * 24 * 7,    # the x-th day of the y-th week of each month
    'LWOM': 60 * 60 * 24 * 7,   # the x-th day of the last week of each month

    # 15 days/half month
    'SM': 60 * 60 * 24 * 15,
    'SME': 60 * 60 * 24 * 15,   # 15th (or other day_of_month) and calendar month end
    'SMS': 60 * 60 * 24 * 15,   # 15th (or other day_of_month) and calendar month begin

    # month
    'M': 60 * 60 * 24 * 30,     # month
    'MS': 60 * 60 * 24 * 30,    # calendar month begin
    'ME': 60 * 60 * 24 * 30,    # calendar month end
    'BM': 60 * 60 * 24 * 30,    # business month
    'BME': 60 * 60 * 24 * 30,   # business month end
    'BMS': 60 * 60 * 24 * 30,   # business month begin
    'CBM': 60 * 60 * 24 * 30,   # custom business month
    'CBME': 60 * 60 * 24 * 30,  # custom business month end
    'CBMS': 60 * 60 * 24 * 30,  # custom business month begin
    'MBS': 60 * 60 * 24 * 30,
    'CMBS': 60 * 60 * 24 * 30,

    # quarter
    'Q': 60 * 60 * 24 * 91,
    'QE': 60 * 60 * 24 * 91,    # calendar quarter end
    'QS': 60 * 60 * 24 * 91,    # calendar quarter start
    'BQ': 60 * 60 * 24 * 91,
    'BQE': 60 * 60 * 24 * 91,   # business quarter end
    'BQS': 60 * 60 * 24 * 91,   # business quarter begin

    # year
    'A': 60 * 60 * 24 * 365,
    'Y': 60 * 60 * 24 * 365,
    'YE': 60 * 60 * 24 * 365,   # calendar year end
    'YS': 60 * 60 * 24 * 365,   # calendar year begin
    'BA': 60 * 60 * 24 * 365,
    'BY': 60 * 60 * 24 * 365,
    'AS': 60 * 60 * 24 * 365,
    'AY': 60 * 60 * 24 * 365,
    'BAE': 60 * 60 * 24 * 365,  # business year end
    'BAS': 60 * 60 * 24 * 365,  # business year begin
    'BYE': 60 * 60 * 24 * 365,  # business year end
    'BYS': 60 * 60 * 24 * 365,  # business year begin

    'REQ': 60 * 60 * 24 * 7 * 52,   # retail (aka 52-53 week) quarter
    'RE': 60 * 60 * 24 * 7 * 52,  # retail (aka 52-53 week) quarter

    # 'L': 1, 'ms': 1,    # milliseconds
    # 'U': 1, 'us': 1,    # microseconds
    # 'N': 1              # nanoseconds

    'WS': 60 * 60 * 24 * 7,  # one week, optionally anchored on a day of the week
    'WE': 60 * 60 * 24 * 7,  # one week, optionally anchored on a day of the week

}


NORMALIZED_FREQ = {
    # second
    'S': 'S',
    's': 'S',

    # minute
    'T': 'min',
    'min': 'min',

    # hour
    'H': 'H',
    'h': 'H',
    'BH': 'H',
    'CBH': 'H',

    # day
    'B': 'D',
    'D': 'D',
    'C': 'D',

    # week
    'W': 'W',       # 7
    'BW': 'BW',     # 5
    'WOM': 'W',     # 7
    'LWOM': 'W',    # 7

    # 15 days/half month
    'SM': 'SM',
    'SME': 'SM',
    'SMS': 'SM',

    # month
    'M': 'M',
    'MS': 'M',
    'ME': 'M',
    'BM': 'M',
    'BME': 'M',
    'BMS': 'M',
    'CBM': 'M',
    'CBME': 'M',
    'CBMS': 'M',
    'MBS': 'M',
    'CMBS': 'M',

    # quarter
    'Q': 'Q',
    'QE': 'Q',
    'QS': 'Q',
    'BQ': 'Q',
    'BQE': 'Q',
    'BQS': 'Q',

    # year
    'A': 'Y',
    'Y': 'Y',
    'YE': 'Y',
    'YS': 'Y',
    'BA': 'Y',
    'BY': 'Y',
    'AS': 'Y',
    'AY': 'Y',
    'BAE': 'Y',
    'BAS': 'Y',
    'BYE': 'Y',
    'BYS': 'Y',

    'REQ': 'RE',
    'RE': 'RE',

}


def normalize_freq(freq):
    # return NORMALIZED_FREQ[freq] if isinstance(freq, str) else freq
    return freq


# ---------------------------------------------------------------------------
# infer_freq
# ---------------------------------------------------------------------------
# Pandas already offer a 'infer_freq' method

def infer_freq(datetime, steps=5, ntries=3) -> str:
    """
    Infer 'freq' checking randomly different positions of the index

    [2024/02/23] implementation simplified using the services offered
                 by pandas.infer_freq
                 It add some extra cases not supported by

    :param datetime: pandas' index to use
    :param steps: number of success results
    :param ntries: maximum number of retries if some check fails
    :return: the inferred frequency
    """
    if isinstance(datetime, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        return pd.infer_freq(datetime)
    elif isinstance(datetime, pd.PeriodIndex):
        return datetime.iloc[0].freqstr
    # other pandas index types
    elif isinstance(datetime, pd.Index):
        return None
    elif isinstance(datetime, pd.Series) and datetime.dtype == pd.PeriodDtype:
        return datetime.iloc[0].freqstr
    elif isinstance(datetime, pd.Period):
        return datetime.freqstr
    else:
        return pd.infer_freq(datetime)

    # freq = None
    # if isinstance(datetime, pd.Period):
    #     freq = datetime.freqstr
    # elif isinstance(datetime, pd.PeriodIndex):
    #     freq = datetime.iloc[0].freqstr
    # elif isinstance(datetime, pd.Series):
    #     dt = datetime.iloc[0]
    #     if hasattr(dt, 'freqstr'):
    #         freq = dt.freqstr
    # if freq is None:
    #     freq = _infer_freq(datetime, steps, ntries)
    #
    # # support: DatetimeIndex | TimedeltaIndex | Series | DatetimeLikeArrayMixin,
    # freq2 = pd.infer_freq(datetime)
    #
    # return NORMALIZED_FREQ[freq]
# end


def _infer_freq(datetime, steps=5, ntries=3):
    n = len(datetime)-steps
    freq = None
    itry = 0
    while itry < ntries:
        i = random.randrange(n)
        tfreq = pd.infer_freq(datetime[i:i+steps])
        if tfreq is None:
            itry += 1
        elif tfreq != freq:
            freq = tfreq
            itry = 0
        else:
            itry += 1
    # end
    return freq
# end
