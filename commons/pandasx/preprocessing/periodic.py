from typing import Union, Optional

import pandas as pd
import numpy as np

from .base import *
from ..base import groups_split, groups_merge

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

PERIOD_LEN = {
    NAME_DAY: 365,
    NAME_WEEK: 52,
    NAME_MONTH: 12,
    NAME_QUARTER: 4,
    NAME_YEAR: 0,
    NAME_DAY_OF_WEEK: 7
}

PERIOD_START = {
    NAME_DAY: 1,
    NAME_WEEK: 1,
    NAME_MONTH: 1,
    NAME_QUARTER: 1,
    NAME_YEAR: 0,
    NAME_DAY_OF_WEEK: 0
}


def _week(ix):
    try:
        return ix.week
    except:
        return ix.isocalendar().week


class Periodic:

    def __init__(self, name, flag, selector):
        self.name = name
        self.flag = flag
        self.selector = selector

    def column_name(self, column: str, suffix=""):
        """
        Compose the column name adding as suffix the period name

        :param column: column name
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
        period_start = PERIOD_START[self.name]
        period_len = PERIOD_LEN[self.name]
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
        period_start = PERIOD_START[self.name]
        period_len = PERIOD_LEN[self.name]
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
    Periodic(NAME_DAY, PERIODIC_DAY, lambda ix: ix.day),
    Periodic(NAME_WEEK, PERIODIC_WEEK, _week),
    Periodic(NAME_DAY_OF_WEEK, PERIODIC_DAY_OF_WEEK, lambda ix: ix.dayofweek),
    Periodic(NAME_MONTH, PERIODIC_MONTH, lambda ix: ix.month),
    Periodic(NAME_QUARTER, PERIODIC_QUARTER, lambda ix: ix.quarter),
    Periodic(NAME_YEAR, PERIODIC_YEAR, lambda ix: ix.year.astype(str)),
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


def _compute_means(df, datetime, periodic, targets):
    means = {}
    for p in PERIODIC_LIST:
        if p.has_period(periodic):
            name = p.name
            means[name] = p.mean(df, datetime, targets)
    return means


def _add_means(df, datetime, periodic, targets, periodic_means):
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

class PeriodicEncoder(BaseEncoder):

    def __init__(self, columns: Union[None, str, list[str]] = None,
                 periodic: int = 0,
                 datetime: Union[None, str, tuple] = None,
                 freq: Optional[str] = None,
                 groups: Union[None, str, list[str]] = None,
                 means=True,
                 periods=None,
                 copy=True):
        """
        Add different periodic types and the means

        If 'target' is None, the means is NOT added
        If 'datetime' is not defined, it is used the index. If defined, it must be a valid datetime column
        If 'groups' is defined, the dataframe is split and merged

        :param columns: column name(s) where to compute the means
        :param periodic: flags of periodics to use
        :param datetime str|(str, str): datetime column. If not specified, it is used the index
        :param freq: datetime frequency
        :param groups: list of colums used to split
        :params means: if to add the means columns
        :params periods: if to add the periods columns
        """
        super().__init__(columns, copy)
        self.periodic = periodic
        self.freq = freq
        self.datetime = datetime

        if datetime is None or isinstance(datetime, str):
            self.datetime = datetime
        else:
            self.datetime = datetime[0]
            self.freq = datetime[1]

        self.groups = as_list(groups)
        self.means = means
        self.periods = periods

        # means for each column and period
        self._means = {}

    def fit(self, df: DataFrame) -> "PeriodicEncoder":
        if len(self.groups) == 0:
            return self._fit(df)

        X_dict: dict = dict()
        groups = groups_split(df, groups=self.groups, drop=True)
        for g in groups:
            X = groups[g]
            X = self._fit(X)
            X_dict[g] = X

        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _fit(self, X: DataFrame) -> "PeriodicEncoder":
        datetime = self.datetime
        targets = self.columns
        periodic = self.periodic

        if len(targets) == 0 or periodic == 0:
            return self

        if datetime is None:
            datetime = NAME_DATETIME
            values = X[targets]
            values[datetime] = values.index.to_series()
        else:
            values = X[targets + [datetime]]

        values = _add_periods(values, datetime, periodic)
        means = _compute_means(values, datetime, periodic, targets)

        self._means = means
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        if len(self.groups) == 0:
            return self._transform(df)

        X_dict: dict = dict()
        groups = groups_split(df, groups=self.groups, drop=True)
        for g in groups:
            X = groups[g]
            X = self._transform(X)
            X_dict[g] = X

        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        datetime = self.datetime
        targets = self.columns
        periodic = self.periodic
        drop_dt = False

        #
        # to simplify the merging of means, it is added
        # a 'virtual' datetime column, removed before
        # to return the dataset

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
        #   3) if necessary, drop the periods
        #

        X = _add_periods(X, datetime, periodic)

        if self.means:
            X = _add_means(X, datetime, periodic, targets, self._means)

        if self.periods in ["fourier", "sincos"]:
            X = _add_fourier(X, datetime, periodic)

        if self.periods in ["oh", "onehot"]:
            X = _add_onehot(X, datetime, periodic)

        if not self.periods:
            X = _drop_periods(X, datetime, periodic)

        if drop_dt:
            X.drop(datetime, axis=1, inplace=True)

        return X
    # end
# end
