from typing import Union, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from stdlib import NoneType

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

FREQ_VALID = ['W', 'M', 'SM', 'BM', 'CBM', 'MS', 'SMS', 'BMS', 'CBMS',
              'Q', 'Q', 'QS', 'BQS',
              'A', 'Y', 'BA', 'BY', 'AS', 'AY', 'BAS', 'BAY']


def _to_datetime(dt_series, format, freq) -> pd.Series:

    def remove_tz(x):
        if hasattr(x, 'tz_localize'):
            return x.tz_localize(None)
        elif isinstance(x, datetime):
            return x.replace(tzinfo=None)
        else:
            return x

    # Note: if 'format' contains the timezone (%z, %Z), it is necessary to normalize
    # the datetime removing it in a 'intelligent' way.
    # The first  solution is to remove the timezone and stop.
    # The second solution is to convert the timestamp in a 'default' timezone (for example UTC)
    # then to remove the TZ reference.
    # Implemented the first one

    if format is not None and freq is not None:
        dt_series = pd.to_datetime(dt_series, format=format)

    elif format is not None:
        dt_series = pd.to_datetime(dt_series, format=format)
        # dt_series = dt_series.map(lambda x: pd.to_datetime(x, format=format))
        # dt_series = dt_series.transform(lambda x: pd.to_datetime(x, format=format))
        # dt_series = dt_series.apply(lambda x: pd.to_datetime(x, format=format))

        if '%z' in format or '%Z' in format:
            # dt_series = dt_series.apply(lambda x: x.tz_convert("UTC").tz_localize(None))
            dt_series = dt_series.apply(remove_tz)

    # np.dtypes.DateTime64DType == "datetime64[ns]"
    # np.dtypes.ObjectDType
    # available in numpy v1.26.4
    #       NOT in numpy v1.24.3
    # try:
    #     if not isinstance(dt_series.dtype, np.dtypes.DateTime64DType):
    #         dt_series = dt_series.astype("datetime64[ns]")
    # except:
    #     pass

    if freq is not None:
        # dt_series = dt_series.dt.to_period(freq)
        dt_series.asfreq(freq)
    return dt_series


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
    assert isinstance(freq, (NoneType, str))
    # assert 1 < len(datetime) < 4
    if isinstance(datetime, str):
        pass
    elif len(datetime) == 1:
        pass
    elif len(datetime) == 2:
        datetime, format = datetime
    else:
        datetime, format, freq = datetime

    dt_series = df[datetime]
    dt_series = _to_datetime(dt_series, format, freq)
    df[datetime] = dt_series

    return df
# end


def datetime_reindex(df: pd.DataFrame, keep='first', mehod='pad') -> pd.DataFrame:
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
