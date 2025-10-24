from datetime import datetime, date
from typing import Literal, Union, Optional

import numpy as np
import pandas as pd

from stdlib import is_instance
from stdlib.dateutilx import relativeperiods
from .base import infer_freq

__all__ = [
    'to_date_type', 'to_datetime', 'to_period', 'date_range',
    'add_date_name'
]

# ---------------------------------------------------------------------------
# to_date_type
# ---------------------------------------------------------------------------

def to_date_type(dt: Union[date, datetime], date_type):
    dt_type = type(dt)
    if dt_type == date_type:
        return dt
    if date_type == pd.Timestamp:
        if dt_type == date:
            return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)
        if dt_type == datetime:
            return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute, second=dt.second)
    return dt


# ---------------------------------------------------------------------------
# to_datetime
# ---------------------------------------------------------------------------

FREQ_VALID = ['W', 'M', 'SM', 'BM', 'CBM', 'MS', 'SMS', 'BMS', 'CBMS',
              'Q', 'Q', 'QS', 'BQS',
              'A', 'Y', 'BA', 'BY', 'AS', 'AY', 'BAS', 'BAY']


def _to_datetime(dt_values, format, freq) -> pd.DatetimeIndex:

    def remove_tz(x):
        if hasattr(x, 'tz_localize'):
            return x.tz_localize(None)
        elif isinstance(x, datetime):
            return x.replace(tzinfo=None)
        else:
            return x

    # Note: if 'format' contains the timezone (%z, %Z), it is necessary to normalize
    # the datetime removing it in an 'intelligent' way.
    # The first  solution is to remove the timezone and stop.
    # The second solution is to convert the timestamp in a 'default' timezone (for example UTC)
    # then to remove the TZ reference.
    # >> Implemented the first one

    if format is not None:
        dt_values = pd.to_datetime(dt_values, format=format)
        dt_values = remove_tz(dt_values)

    elif format is not None:
        dt_values = pd.to_datetime(dt_values, format=format)

        if '%z' in format or '%Z' in format:
            # dt_series = dt_series.apply(lambda x: x.tz_convert("UTC").tz_localize(None))
            dt_values = dt_values.apply(remove_tz)
    else:
        dt_values = pd.to_datetime(dt_values)

    return dt_values


NP_FIRST_JAN_1970 = np.datetime64('1970-01-01T00:00:00')
NP_ONE_SECOND = np.timedelta64(1, 's')


def to_datetime(dt, format=None, freq=None) \
        -> Union[datetime, pd.DatetimeIndex]:
    """
    Try to convert into a datetime an object represented in several formats

    :param dt:  object to convert
    :return: datetime
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        return pd.to_datetime(dt).to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, pd.Period):
        return dt.to_timestamp().to_pydatetime()
    if isinstance(dt, np.datetime64):
        ts = ((dt - NP_FIRST_JAN_1970) / NP_ONE_SECOND)
        dt = datetime.utcfromtimestamp(ts)
        return dt
    else:
        raise ValueError(f"Unsupported datetime {dt}")


# ---------------------------------------------------------------------------
# to_period
# ---------------------------------------------------------------------------

def to_period(dt: Union[str, list[str], datetime], *, format=None, freq=None) -> Union[pd.Period, pd.PeriodIndex]:
    if isinstance(dt, str):
        dt = to_datetime(dt)
        return pd.Period(dt, freq=freq)
    elif isinstance(dt, list):
        dt: pd.DatetimeIndex = pd.to_datetime(dt, format=format)
        if freq is None:
            freq = infer_freq(dt)
        return dt if freq is None else dt.to_period(freq)
    elif isinstance(dt, pd.DatetimeIndex):
        if freq is None:
            freq = infer_freq(dt)
        return dt if freq is None else dt.to_period(freq)
    else:
        return pd.Period(dt, freq=freq)
# end


# ---------------------------------------------------------------------------
# date_range
# ---------------------------------------------------------------------------

INCLUSIVE_TYPE = Literal['left', 'right', 'both', 'neither']


def _sdp(start_date: datetime, periods: int, delta) -> list[datetime]:

    dr = []
    cdt = start_date
    for p in range(periods):
        dr.append(cdt)
        cdt += delta

    return dr


def _edp(end_date: datetime, periods: int, delta) -> list[datetime]:

    dr = []
    cdt = end_date
    for p in range(periods):
        dr.append(cdt)
        cdt -= delta

    dr = dr[::-1]
    return dr


def _sedl(start_date, end_date, delta) -> list[datetime]:
    # start_date, end_date, left

    dr = []
    cdt = start_date
    ldt = start_date
    while cdt <= end_date:
        dr.append(cdt)
        ldt = cdt
        cdt += delta
    if ldt != end_date:
        dr.append(cdt)

    return dr


def _sedr(start_date, end_date, delta) -> list[datetime]:
    # start_date, end_date, right

    dr = []
    cdt = end_date
    ldt = end_date
    while cdt >= start_date:
        dr.append(cdt)
        ldt = cdt
        cdt -= delta
    if ldt != start_date:
        dr.append(cdt)

    dr = dr[::-1]
    return dr


def date_range(start=None, end=None, periods=None,
               freq=None, delta=None,
               name=None,
               inclusive: INCLUSIVE_TYPE = 'both', align='left') -> pd.DatetimeIndex:
    """
    This function has a behavior very similar than 'pd.date_range', with the following difference:
    it is possible to specify timestamp with a specified freq)ency, starting at each datetime.
    For example, it is possible to generate weekly timestamps starting from Wednesday, not from
    Sunday or Monday.

    It is also possible to generate sequences of timestamps starting from 'start_date' or 'end_date'
    Possibilities:

        start_date  periods     freq            inclusive       (forward)
        end_date    periods     freq            inclusive       (backward)
        start_date  end_date    freq    align   inclusive       (left:forward, right:backward)

    It is possible to include or exclude the time limits: 'both', 'left', 'right', 'neither'

    :param start: start date
    :param end: end date
    :param periods: n of periods
    :param freq: frequency
    :param delta: alternative to (periods, freq)
    :param name: name of the series
    :param inclusive: if to include/exclude the extreme ('left', 'right', 'both', 'neither')
    :param align: if to aligh the date 'left' or 'right'
    :return:
    """
    assert isinstance(freq, (type(None), str))
    assert delta is not None or freq is not None

    start_date = to_datetime(start)
    end_date = to_datetime(end)

    if delta is None:
        delta = relativeperiods(periods=1, freq=freq)

    if periods is not None:
        if inclusive in ['neither', 'right']:
            periods += 1
        if inclusive in ['neither', 'left']:
            periods += 1
    elif start_date is not None and end_date is not None:
        if inclusive in ['neither', 'right']:
            # start_date += delta
            pass
        if inclusive in ['neither', 'left']:
            # end_date -= delta
            pass
    else:
        raise ValueError(f"Unsupported start/end/periods combination {start, end, periods}")

    if start_date is not None and end_date is not None:
        if align == 'left':
            dr = _sedl(start_date, end_date, delta)
        elif align == 'right':
            dr = _sedr(start_date, end_date, delta)
        else:
            raise ValueError(f"Unsupported alignment {align}")
    elif start_date is not None and periods is not None:
        dr = _sdp(start_date, periods, delta)
    elif end_date is not None and periods is not None:
        dr = _edp(end_date, periods, delta)
    else:
        raise ValueError(f"Unsupported start/end/periods combination {start, end, periods}")

    if inclusive in ['neither', 'right']:
        dr = dr[1:]
    if inclusive in ['neither', 'left']:
        dr = dr[:-1]

    return pd.DatetimeIndex(data=dr, name=name)
# end



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


# ---------------------------------------------------------------------------
# add_date_name
# ---------------------------------------------------------------------------

def add_date_name(df: pd.DataFrame, dateCol: str, freq: Literal['D', 'W', 'M'], dowCol: Optional[str] = None) \
        -> tuple[pd.DataFrame, str]:
    assert is_instance(freq, Literal['D', 'W', 'M'])

    if dowCol is not None:
        pass
    elif freq in ['D']:
        dowCol = 'dow'
    elif freq in ['W']:
        dowCol = 'moy'
    elif freq in ['M']:
        dowCol = 'moy'
    else:
        raise ValueError(f"Unsupported freq {freq}")

    if dowCol in df.columns:
        return df, dowCol

    if freq in ['D']:
        df[dowCol] = df[dateCol].dt.day_name()
    elif freq in ['W']:
        # the name of the day is not very useful because it is the same!
        # an alternative is the name of the month
        df[dowCol] = df[dateCol].dt.month_name()
    elif freq in ['M']:
        df[dowCol] = df[dateCol].dt.month_name()
    else:
        raise ValueError(f"Unsupported freq {freq}")

    return df, dowCol
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
