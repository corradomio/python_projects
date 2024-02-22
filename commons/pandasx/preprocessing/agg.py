from typing import Union
from pandas import DataFrame, Series, Index
from stdlib import NoneType, is_instance, lrange

from .base import GroupsEncoder
from ..freq import FREQUENCIES, normalize_freq, infer_freq


# ---------------------------------------------------------------------------
# DatetimeInfo
# ColumnInfo
# ---------------------------------------------------------------------------

class DatetimeInfo:
    def __init__(self, datetime):
        assert isinstance(datetime, Series)

        # Period:
        #   day, month, quarter, week
        #   day, dayofweek, dayofyear, weekday
        #   daysinmonth
        #   freq, freqstr

        self.datetime= datetime
        self.freq = infer_freq(datetime)
        pass


class ColumnInfo:
    def __init__(self, column):
        assert isinstance(column, Series)
        self.min = column.min()
        self.max = column.max()
        self.sum = column.sum()
        self.ratio = column.values/self.sum if self.sum != 0 else column.values
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# problem:
#   which is the first day of the week? the first work day OR the last weekend day?
#   which are the work week days?  [monday-friday]/[sunday-thursday]/...
#   which are the weekend days?    [saturday-sunday]/[friday-saturday]/...

# select datetime: for now we suppose that datetime is
#   daily           all days
#   work daily      only the work days
#   weekend         the last 2 days or the week
#
# Aggregate daily -> weekly
#   5 or 7 days?    this is automatic depends on the dataset
#   which day of the week to use?
#       first day of the week
#       last  day of the week   <-- default
#       first work day
#       last  work day
#       first weekend day
#       last  weekend day
#
# Aggregate daily -> monthly
#   which day of the month to use?
#       first day               <-- default
#       last  day
#       first work day
#       last  work day
#       first weekend day
#       last  weekend day

def _select_each(series: Series, offset, step) -> Series:
    n = len(series)
    indices = lrange(offset, n, step)
    return series.iloc[indices]


def _select_on(series: Series, offset, step) -> Series:
    # each week
    if step == 'W':
        pass
    # each month
    elif step == 'M':
        pass


def _aggregate_each(series: Series, offset, step, index) -> Series:
    agg_values = []
    for i in range(offset, len(series), step):
        agg_values.append(series.iloc[i:i+step].sum())
    agg_values = Series(data=agg_values, index=index)
    return agg_values

# ---------------------------------------------------------------------------
# AggregateTransformer
# ---------------------------------------------------------------------------
#   Pandas Index
#       Index
#           RangeIndex          Index implementing a monotonic integer range.
#           CategoricalIndex    Index of Categorical's.
#           MultiIndex          A multi-level, or hierarchical Index.
#           IntervalIndex       An Index of Interval's.
#           DatetimeIndex       Index of datetime64 data.
#           TimedeltaIndex      Index of timedelta64 data.
#           PeriodIndex         Index of Period data.
#
# Aggregate information:
#   data start_date - end_date,
#   data period_length

class AggregateTransformer(GroupsEncoder):
    """

    """

    def __init__(self, columns=None,
                 freq: Union[None, str, int]=None,
                 align: Union[None, str, int]=None,
                 datetime=None, groups=None,
                 copy=True):
        """

        :param columns: columns to aggregate
        :param datetime: datetime columns. If specified, the columns must be compatible 'datetime' value
                if the columns doesn't contain the information about the 'frequency'/period, it is considered
                single value
        :param freq: aggregated frequency. Can be pandas frequency string or an integer value
        :param align: data alignment
        :param groups: columns used to identify the TS in a multi-TS dataset
                If None, it is used the MultiIndex
        :param copy:
        """
        assert is_instance(freq, Union[NoneType, int, str])
        assert isinstance(freq, int) or freq in FREQUENCIES, f"Unknown 'freq' value: {freq}"
        super().__init__(columns, groups, copy)
        self.datetime = datetime
        self.freq = normalize_freq(freq)
        self.align = align
        self._datetime_infos = None
        self._infos = {}

    # -----------------------------------------------------------------------

    def _check_X(self, X: DataFrame):
        if len(self.columns) == 0:
            self.columns = X.columns
        return super()._check_X(X)

    # -----------------------------------------------------------------------

    def _get_params(self, g):
        infos = self._infos[g]
        return infos

    def _set_params(self, g, params):
        infos = params
        self._infos[g] = infos
        pass

    def _compute_params(self, X):
        # retrieve the information about the
        infos = dict()
        self._get_datetime_info(X)
        for col in self.columns:
            if col == self.datetime:
                continue
            infos[col] = ColumnInfo(X[col])
        return infos

    def _get_datetime_info(self, X):
        if self._datetime_infos is not None:
            return self._datetime_infos

        if self.datetime is None:
            datetime = X.index.to_series()
        else:
            datetime = X[self.datetime]

        self._datetime_infos = DatetimeInfo(datetime)
        return self._datetime_infos

    def _apply_transform(self, X, params):
        infos = params
        agg_datetime = self._aggregate_datetime()
        agg_columns = {}
        for col in self.columns:
            agg_columns[col] = self._aggregate_column(X[col], agg_datetime)

        agg_df = DataFrame(data=agg_columns, index=agg_datetime)
        return agg_df

    def _aggregate_datetime(self) -> Series:
        freq_agg = (self._datetime_infos.freq, self.freq)
        if isinstance(freq_agg[1], int):
            agg_dt = _select_each(self._datetime_infos.datetime, 0, self.freq)
        elif freq_agg == ('D', 'W'):
            agg_dt = _select_on(self._datetime_infos.datetime, 0, self.freq)
        elif freq_agg == ('D', 'M'):
            agg_dt = _select_on(self._datetime_infos.datetime, 0, self.freq)
        else:
            raise ValueError(f'Unsupported frequency aggregation {freq_agg[0]}->{freq_agg[1]}')
        return agg_dt

    def _aggregate_column(self, column, agg_datetime) -> Series:
        # frequency aggregation
        freq_agg = (self._datetime_infos.freq, self.freq)
        if isinstance(freq_agg[1], int):
            agg_col = _aggregate_each(column, 0, self.freq, agg_datetime)
        elif freq_agg == ('D', 'W'):
            pass
        elif freq_agg == ('D', 'M'):
            pass
        else:
            raise ValueError(f'Unsupported frequency aggregation {freq_agg[0]}->{freq_agg[1]}')
        return agg_col

    def _apply_inverse_transform(self, X, params):
        pass
# end

