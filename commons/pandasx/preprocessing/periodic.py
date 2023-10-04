from typing import Union, Optional

import numpy as np
import pandas as pd

from stdlib import as_list
from .base import XyBaseEncoder
from ..base import groups_split, groups_merge


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERIODIC_DAY = 0x0001
PERIODIC_WEEK = 0x0002
PERIODIC_MONTH = 0x0004
PERIODIC_QUARTER = 0x0008
PERIODIC_YEAR = 0x0010
PERIODIC_DAY_OF_WEEK = 0x0100
PERIODIC_ALL = 0xFFFF
PERIODIC_DMY = PERIODIC_DAY | PERIODIC_MONTH | PERIODIC_YEAR

# PERIODIC_WEEK_OF_MONTH = 0x0200
# PERIODIC_WEEK_OF_YEAR = 0x0400

NAME_DAY = "_D"
NAME_WEEK = "_W"
NAME_MONTH = "_M"
NAME_QUARTER = "_Q"
NAME_YEAR = "_Y"
NAME_DAY_OF_WEEK = "_DOW"

# NAME_WEEK_OF_MONTH = "wom"
# NAME_WEEK_OF_YEAR = "woy"


def _week(ix):
    try:
        return ix.week
    except:
        return ix.isocalendar().week


class Periodic:

    def __init__(self, name, flag, selector, start, len):
        self.name = name
        self.flag = flag
        self.selector = selector
        self.start = start
        self.len = len

    def column_name(self, column: str, suffix=""):
        """
        Compose the column name adding as suffix the period name

        :param column: column name
        :param suffix: column name suffix
        :return: the new name
        """
        return f"{column}{self.name}{suffix}"

    def has_period(self, periodic):
        """
        Check if periodic contains the current period

        :param periodic: periodic falgs
        :return: True if periodic contains the current period
        """
        return periodic & self.flag

    def add_period(self, df, ix, datetime):
        """
        Add the column with the periodic value

        :param df: dataframe to update
        :param ix: PeriodIndex containing the datetime
        :param datetime: name of the column
        :return:
        """
        col = self.column_name(datetime)
        df[col] = self.selector(ix)

    def mean(self, df, datetime, targets):
        """
        Compute the mean for the specified period

        :param df: dataframe containing the data
        :param datetime: name of the column
        :param targets: name of the target columns
        :return:
        """
        col = self.column_name(datetime)
        means = df[[col] + targets].groupby([col]).mean(targets).reset_index()
        return means

    def add_fourier(self, df, ix, datetime):
        period_start = self.start
        period_len = self.len
        if period_len == 0:
            return

        scol = self.column_name(datetime, "_S")
        ccol = self.column_name(datetime, "_C")

        period = (self.selector(ix).to_numpy() - period_start)/period_len
        psin = np.sin(period*2*np.pi)
        pcos = np.cos(period*2*np.pi)

        df[scol] = psin
        df[ccol] = pcos

    def add_onehot(self, df, ix, datetime):
        period_start = self.start
        period_len = self.len
        if period_len == 0:
            return

        for i in range(period_len):
            pi = period_start + i
            icol = self.column_name(datetime, f"_{pi}")
            df[icol] = 0

        period = self.selector(ix).to_numpy()
        for i in range(period_len):
            pi = period_start + i
            icol = self.column_name(datetime, f"_{pi}")
            dfc = df[icol]
            dfc[period == pi] = 1
            df[icol] = dfc
            pass


PERIODIC_LIST = [
    Periodic(NAME_DAY, PERIODIC_DAY, lambda ix: ix.day, 1, 365),
    Periodic(NAME_WEEK, PERIODIC_WEEK, _week, 1, 52),
    Periodic(NAME_MONTH, PERIODIC_MONTH, lambda ix: ix.month, 1, 12),
    Periodic(NAME_QUARTER, PERIODIC_QUARTER, lambda ix: ix.quarter, 1, 4),
    Periodic(NAME_YEAR, PERIODIC_YEAR, lambda ix: ix.year.astype(str), 0, 0),
    Periodic(NAME_DAY_OF_WEEK, PERIODIC_DAY_OF_WEEK, lambda ix: ix.dayofweek, 0, 7),
]

NAME_DATETIME = "$DT"

# composite
#
#   quarter_year    'quarter'_'year'
#   month_week      'month'_'week'
#   week_year       'week'_'year'
#   weekly_monthly  'day'_'week'_'month'
#


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_index(df, datetime):
    """
    Convert df[datetime] in a PeriodIndex

    :param df:
    :param datetime: column_name or (column_name, freq)
    :return:
    """
    freq = None
    if isinstance(datetime, (list, tuple)):
        datetime, freq = datetime
    if datetime:
        dt = df[datetime]
        ix = pd.PeriodIndex(data=dt, freq=freq)
    else:
        ix = df.index
    return ix


def _add_periods(df, datetime, periodic):
    ix: pd.PeriodIndex = pd.PeriodIndex(data=df[datetime])
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            p.add_period(df, ix, datetime)
    return df


def _drop_periods(df, datetime, periodic):
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            dtname = p.column_name(datetime)
            df.drop(dtname, axis=1, inplace=True)
    return df


def _compute_means(df, datetime, periodic, targets=None):
    means = {}
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            name = p.name
            means[name] = p.mean(df, datetime, targets)
    return means


def _add_means(df, datetime, periodic, periodic_means, targets):
    ix = df.index
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            dtname = p.column_name(datetime)
            name = p.name
            means = periodic_means[name]

            for target in targets:
                periodic_target = p.column_name(target)
                target_means = means[[dtname, target]]
                target_means.columns = [dtname, periodic_target]
                df = pd.merge(df, target_means, how='left', on=dtname, copy=False)
    # end
    df.index = ix
    return df


def _add_fourier(df, datetime, periodic):
    ix: pd.PeriodIndex = pd.PeriodIndex(data=df[datetime])
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            p.add_fourier(df, ix, datetime)
    return df


def _add_onehot(df, datetime, periodic):
    ix: pd.PeriodIndex = pd.PeriodIndex(data=df[datetime])
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            p.add_onehot(df, ix, datetime)
    return df


# ---------------------------------------------------------------------------
# PeriodicEncoder
# ---------------------------------------------------------------------------
# Problem: if the dataset contains multiple time series, we have 2 solutions
#
#   1) ds contains groups + features + targets
#   2) there are 2 datasets, one for X and one for y, but BOTH
#      must contain the group columns!
#

class PeriodicEncoder(XyBaseEncoder):

    def __init__(self,
                 datetime: Union[None, str, tuple] = None,
                 *,
                 periodic: int = 0,
                 freq: Optional[str] = None,
                 groups: Union[None, str, list[str]] = None,
                 means=True,
                 method=None,
                 copy=True):
        """
        Add different periodic columns and the means

        If 'datetime' is not defined, it is used the index. If defined, it must be a valid datetime column
        If 'groups' is defined, the transformation is applied to each group

        :param datetime str|(str, str): datetime column. If not specified, it is used the index
        :param freq: datetime frequency
        :param periodic: flags of periodics to use
        :param groups str|list[str]: list of columns used to split the sub time series
        :params means bool: if to add the means columns for each period
        :params method str: if to add the periods columns:
                None: no period columns are added
                "oh", "onehot": add columns in onehot encoding
                "fourier", "sincos": add columns in fourier form
        """
        super().__init__(None, copy)
        self.periodic = periodic
        self.freq = freq

        if datetime is None or isinstance(datetime, str):
            self.datetime = datetime
        else:
            self.datetime = datetime[0]
            self.freq = datetime[1]

        self.groups = as_list(groups)
        self.means = means
        self.method = method

        # means for each column and period
        self._means = {}
        self._targets = None

    def fit(self, X, y) -> "PeriodicEncoder":
        if not self.means:
            return self

        self._check_datetime(X, y)

        if len(self.groups) == 0:
            y = y.copy()
            self._targets = list(y.columns)
            means = self._compute_means(y)
            self._means = means
            return self

        groups = groups_split(y, groups=self.groups, drop=True)
        for g in groups:
            yg = groups[g]
            self._targets = list(yg.columns)
            means = self._compute_means(yg)
            self._means[g] = means

        return self

    def _compute_means(self, y):
        datetime = self.datetime
        periodic = self.periodic
        targets = list(y.columns.difference([datetime]))
        values = y

        if datetime is None:
            datetime = NAME_DATETIME
            values[datetime] = values.index.to_series()
        elif datetime not in values.columns:
            values[datetime] = values.index.to_series()

        values = _add_periods(values, datetime, periodic)
        means = _compute_means(values, datetime, periodic, targets=targets)

        return means

    def _check_datetime(self, X, y):
        datetime = self.datetime
        if datetime is not None:
            assert X is None or datetime in X.columns, f"The column {datetime} is not present in X"
            assert y is None or datetime in y.columns, f"The column {datetime} is not present in y"

    def transform(self, X=None, y=None) -> pd.DataFrame:
        if len(self.groups) == 0:
            return self._transform(X, self._means)

        X_dict: dict = dict()
        groups = groups_split(X, groups=self.groups, drop=True) if X is not None else None

        for g in groups:
            Xg = groups[g]
            means = self._means[g]
            Xg = self._transform(Xg, means)
            X_dict[g] = Xg

        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _transform(self, X, means) -> pd.DataFrame:
        X = self._check_X(X)
        X = X.copy()

        datetime = self.datetime
        targets = self._targets
        periodic = self.periodic
        drop_dt = False
        drop_periods = self.method is None

        #
        # to simplify the merging of means, it is added
        # a 'virtual' datetime column, removed before
        # to return the dataset
        #

        if datetime is None:
            datetime = NAME_DATETIME
            X[datetime] = X.index.to_series()
            drop_dt = True

        #
        # it is necessary to encode the periods to add means correctly.
        # Then:
        #
        #   1) add the periods
        #   2) if necessary, add the means
        #   3) if necessary, add the periods
        #   3) if necessary, drop the periods
        #

        X = _add_periods(X, datetime, periodic)

        if self.means:
            X = _add_means(X, datetime, periodic, means, targets)

        if self.method in [True, "plain"]:
            pass

        if self.method in ["fourier", "sincos"]:
            X = _add_fourier(X, datetime, periodic)
            drop_periods = True

        if self.method in ["oh", "onehot"]:
            X = _add_onehot(X, datetime, periodic)
            drop_periods = True

        if drop_periods:
            X = _drop_periods(X, datetime, periodic)

        if drop_dt:
            X.drop(datetime, axis=1, inplace=True)

        return X
    # end
# end

# class PeriodicEncoder(BaseEncoder):
#
#     def __init__(self,
#                  columns: Union[None, str, list[str]] = None,
#                  periodic: int = 0,
#                  datetime: Union[None, str, tuple] = None,
#                  freq: Optional[str] = None,
#                  groups: Union[None, str, list[str]] = None,
#                  means=True,
#                  method=None):
#         """
#         Add different periodic types and the means
#
#         If 'datetime' is not defined, it is used the index. If defined, it must be a valid datetime column
#         If 'groups' is defined, the dataframe is split and merged
#
#         :param columns: column name(s) where to compute the means
#         :param periodic: flags of periodics to use
#         :param datetime str|(str, str): datetime column. If not specified, it is used the index
#         :param freq: datetime frequency
#         :param groups: list of columns used to split the sub time series
#         :params means: if to add the means columns for each period
#         :params method: if to add the periods columns:
#             None: no period columns are added
#             "oh", "onehot": add columns in onehot encoding
#             "fourier", "sincos": add columns in fourier form
#         """
#         super().__init__(columns, False)
#         self.periodic = periodic
#         self.freq = freq
#         self.datetime = datetime
#
#         if datetime is None or isinstance(datetime, str):
#             self.datetime = datetime
#         else:
#             self.datetime = datetime[0]
#             self.freq = datetime[1]
#
#         self.groups = as_list(groups)
#         self.means = means
#         self.method = method
#
#         # means for each column and period
#         self._means = {}
#
#     def fit(self, df: pd.DataFrame) -> "PeriodicEncoder":
#         if not self.means:
#             return self
#
#         if len(self.groups) == 0:
#             X = df
#             means = self._compute_means(X)
#             self._means = means
#             return self
#
#         groups = groups_split(df, groups=self.groups, drop=True)
#         for g in groups:
#             X = groups[g]
#             means = self._compute_means(X)
#             self._means[g] = means
#
#         return self
#
#     def _compute_means(self, X: pd.DataFrame):
#         datetime = self.datetime
#         targets = self.columns
#         periodic = self.periodic
#
#         if len(targets) == 0 or periodic == 0:
#             return self
#
#         if datetime is None:
#             datetime = NAME_DATETIME
#             values = X[targets]
#             values[datetime] = values.index.to_series()
#         else:
#             values = X[targets + [datetime]]
#
#         values = _add_periods(values, datetime, periodic)
#         means = _compute_means(values, datetime, periodic, targets)
#
#         return means
#
#     def transform(self, df: pd.DataFrame) -> pd.DataFrame:
#         if len(self.groups) == 0:
#             return self._transform(df, self._means)
#
#         X_dict: dict = dict()
#         groups = groups_split(df, groups=self.groups, drop=True)
#         for g in groups:
#             X = groups[g]
#             means = self._means[g]
#             X = self._transform(X, means)
#             X_dict[g] = X
#
#         X = groups_merge(X_dict, groups=self.groups)
#         return X
#
#     def _transform(self, X: pd.DataFrame, means) -> pd.DataFrame:
#         X = self._check_X(X)
#
#         datetime = self.datetime
#         targets = self.columns
#         periodic = self.periodic
#         drop_dt = False
#         drop_periods = self.method is None
#
#         #
#         # to simplify the merging of means, it is added
#         # a 'virtual' datetime column, removed before
#         # to return the dataset
#         #
#
#         if datetime is None:
#             datetime = NAME_DATETIME
#             X[datetime] = X.index.to_series()
#             drop_dt = True
#
#         #
#         # it is necessary to encode the periods to add means correctly.
#         # Then:
#         #
#         #   1) add the periods
#         #   2) if necessary, add the means
#         #   3) if necessary, add the periods
#         #   3) if necessary, drop the periods
#         #
#
#         X = _add_periods(X, datetime, periodic)
#
#         if self.means:
#             X = _add_means(X, datetime, periodic, targets, means)
#
#         if self.method in [True, "plain"]:
#             pass
#
#         if self.method in ["fourier", "sincos"]:
#             X = _add_fourier(X, datetime, periodic)
#             drop_periods = True
#
#         if self.method in ["oh", "onehot"]:
#             X = _add_onehot(X, datetime, periodic)
#             drop_periods = True
#
#         if drop_periods:
#             X = _drop_periods(X, datetime, periodic)
#
#         if drop_dt:
#             X.drop(datetime, axis=1, inplace=True)
#
#         return X
#     # end
# # end


# ---------------------------------------------------------------------------
