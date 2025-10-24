from typing import Union, Optional, cast

import pandas as pd

from stdlib import NoneType

__all__ = ['datetime_encode', 'datetime_reindex']

# ---------------------------------------------------------------------------
# Datetime encoding:
#   There are several services to support
#
#   1) string:  datetime object DatetimeIndex
#               period object   PeriodIndex
#   2) add periodicity
#       hourly  (0-23)      onehot, sincos
#       weekly  (0-6)       onehot, sincos
#       monthly (0-11)      onehot, sincos
#       last_week_in_month  [0,1]
#


# ---------------------------------------------------------------------------
# datetime_encode
# datetime_reindex
# ---------------------------------------------------------------------------

def datetime_encode(df: pd.DataFrame,
                    datetime: Union[str, tuple[str]],
                    format: Optional[str] = None,
                    freq: Optional[str] = None) -> pd.DataFrame:
    """
    Convert a string column in datatime/period, based on pandas' 'to_datetime' (pd.to_datetime)

    :param df: dataframe to process
    :param datetime: col | (col, format) | (col, format, freq)
    :param format: datetime format
    :param freq: period frequency
    :return: the df with the 'datetime' column converted
    """
    assert isinstance(datetime, (str, list, tuple))
    assert isinstance(format, (NoneType, str))
    # assert isinstance(freq, (NoneType, str))
    # assert 1 < len(datetime) < 4
    if isinstance(datetime, str):
        pass
    elif len(datetime) == 1:
        datetime = datetime[0]
    elif len(datetime) == 2:
        datetime, format = cast(list, datetime)
    else:
        datetime, format, freq = cast(list, datetime)

    dt_series = df[datetime]
    dt_series = pd.to_datetime(list(dt_series), format=format)
    # if freq is None:
    #     freq = infer_freq(dt_series)
    # if freq is not None:
    #     dt_series = dt_series.to_period(freq)
    df[datetime] = dt_series

    return df
# end


def datetime_reindex(df: pd.DataFrame, keep='first', method='pad') -> pd.DataFrame:
    """
    Make sure that the datetime index in dataframe is complete, based
    on the index's 'frequency'
    :param df: dataframe to process
    :param keep: used in 'index.duplicated(leep=...)'
    :param method: used in 'index.reindex(method=...)'
    :return: reindexed dataframe
    """
    start = df.index[0]
    dtend = df.index[-1]
    freq = start.freq
    dtrange = pd.period_range(start, dtend+1, freq=freq)
    # remove duplicated index keys
    df = df[~df.index.duplicated(keep=keep)]
    # make sure that all timestamps are present
    df = df.reindex(index=dtrange, method=mehod)
    return df
# end

# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
