# WARNING: DOESN'T remove 'DataFrame'
from pandas import DataFrame, DatetimeIndex, TimedeltaIndex, PeriodIndex, MultiIndex
from pandas._typing import *

from stdlib import as_list, kwval
from .base import groups_split, groups_merge, index_split, index_merge


# ---------------------------------------------------------------------------
# pandas resample
# ---------------------------------------------------------------------------
# DataFrame.resample(
#   rule,           DateOffset, Timedelta or str    The offset string or object representing target conversion.
#   axis,           {0 or ‘index’, 1 or ‘columns’}, default 0
#                   Which axis to use for up- or down-sampling. For Series this parameter is unused and defaults to 0.
#                   Must be DatetimeIndex, TimedeltaIndex or PeriodIndex.
#                   Deprecated since version 2.0.0: Use frame.T.resample(…) instead.
#   closed,         {‘right’, ‘left’}, default None
#                   Which side of bin interval is closed. The default is ‘left’ for all frequency offsets except for
#                   ‘ME’, ‘YE’, ‘QE’, ‘BME’, ‘BA’, ‘BQE’, and ‘W’ which all have a default of ‘right’.
#   label,          {‘right’, ‘left’}, default None
#                   Which bin edge label to label bucket with. The default is ‘left’ for all frequency offsets except
#                   for ‘ME’, ‘YE’, ‘QE’, ‘BME’, ‘BA’, ‘BQE’, and ‘W’ which all have a default of ‘right’.
#   convention,     {‘start’, ‘end’, ‘s’, ‘e’}, default ‘start’
#                   For PeriodIndex only, controls whether to use the start or end of rule.
#                   Deprecated since version 2.2.0: Convert PeriodIndex to DatetimeIndex before resampling instead.
#   kind,           {‘timestamp’, ‘period’}, optional, default None
#                   Pass ‘timestamp’ to convert the resulting index to a DateTimeIndex or ‘period’ to convert it to a
#                   PeriodIndex. By default the input representation is retained.
#                   Deprecated since version 2.2.0: Convert index to desired type explicitly instead.
#   on,             str, optional
#                   For a DataFrame, column to use instead of index for resampling. Column must be datetime-like.
#   level,          str or int, optional
#                   For a MultiIndex, level (name or number) to use for resampling. level must be datetime-like.
#   origin,         Timestamp or str, default ‘start_day’
#                   The timestamp on which to adjust the grouping. The timezone of origin must match the timezone of
#                   the index. If string, must be one of the following:
#                       ‘epoch’: origin is 1970-01-01
#                       ‘start’: origin is the first value of the timeseries
#                       ‘start_day’: origin is the first day at midnight of the timeseries
#                       ‘end’: origin is the last value of the timeseries
#                       ‘end_day’: origin is the ceiling midnight of the last day
#   offset,         Timedelta or str, default is None
#                   An offset timedelta added to the origin.
#   group_keys      bool, default False
#                   Whether to include the group keys in the result index when using .apply() on the resampled object.
# )
#


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------
#       axis: Axis | lib.NoDefault = lib.no_default,
#       closed: Literal["right", "left"] | None = None,
#       label: Literal["right", "left"] | None = None,
#       convention: Literal["start", "end", "s", "e"] = "start",
#       kind: Literal["timestamp", "period"] | None = None,
#       on: Level | None = None,
#       level: Level | None = None,
#       origin: str | TimestampConvertibleTypes = "start_day",
#       offset: TimedeltaConvertibleTypes | None = None,
#       group_keys: bool_t = False,

METHODS = {
    'sum': lambda x: x.sum(),
    'min': lambda x: x.min(),
    'max': lambda x: x.max(),
    'mean': lambda x: x.mean(),
    'sdev': lambda x: x.sdev(),
}


def _resample_single(df, rule, method, **kwargs):
    resampled = df.resample(rule, **kwargs)
    return method(resampled)


def _resample_multiindex(df, rule, method, **kwargs):
    names = df.index.names
    dfdict = index_split(df)
    drdict = {}
    for g in dfdict:
        dfg = dfdict[g]
        drg = _resample_single(dfg, rule, method, **kwargs)
        drdict[g] = drg

    dr = index_merge(drdict, names=names)
    return dr


def _resample_groups(df, groups: list[str], rule, method, **kwargs):
    dfdict = groups_split(df, groups=groups)
    drdict = {}
    for g in dfdict:
        dfg = dfdict[g]
        drg = _resample_single(dfg, rule, method, **kwargs)
        drdict[g] = drg

    dr = groups_merge(drdict, groups=groups)
    return dr
# end


def resample(df: DataFrame,
             rule,
             method='sum',
             groups: Union[None, str, list[str]]=None,
             **kwargs
             ):
    """

    :param df:      dataframe to process
    :param rule:    aggregation frequency
    :param method:  method to apply to compute the aggregate value ('sum', 'min', 'max', 'mean')
    :param groups:  columns to use for grouping
    :param kwargs:  named arguments passed to 'DataFrame.resample()'
    :return: the resampled DataFrame
    """
    assert isinstance(df, DataFrame), f"Unsupported type: {type(df)}"
    if isinstance(method, str) and not method in METHODS:
        raise ValueError(f'Unknown method {method}')

    method = METHODS[method] if isinstance(method, str) else method
    groups = as_list(groups, 'groups')
    index = df.index
    level = kwval(kwargs, 'level', None)

    if isinstance(index, (DatetimeIndex, TimedeltaIndex or PeriodIndex)) and len(groups) == 0:
        return _resample_single(df, rule, method, **kwargs)
    if isinstance(index, MultiIndex) and level is not None:
        return _resample_single(df, rule, method, **kwargs)
    if isinstance(index, MultiIndex):
        return _resample_multiindex(df, rule, method, **kwargs)
    if len(groups) > 0:
        return _resample_groups(df, rule, method, **kwargs)
    else:
        return _resample_single(df, rule, method, **kwargs)

# end


