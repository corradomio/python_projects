from typing import Union, Optional
from stdlib import NoneType

import pandas as pd

from .base import *

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

    def __init__(self, name, flag, selector):
        self.name = name
        self.flag = flag
        self.selector = selector

    def column_name(self, column: str):
        """
        Compose the column name adding as suffix the period name

        :param column: column name
        :return: the new name
        """
        return f"{column}{self.name}"

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

    :param df:
    :param datetime: column_name or (column_name, freq)
    :return:
    """
    freq = None
    if isinstance(datetime, (list, tuple)):
        datetime, freq = datetime
    dt = df[datetime]
    ix = pd.PeriodIndex(data=dt, freq=freq)
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


# ---------------------------------------------------------------------------
# PeriodicEncoder
# ---------------------------------------------------------------------------

class PeriodicEncoder(BaseEncoder):

    def __init__(self, periodic: int = 0,
                       target: Union[None, str, list[str]] = None,
                       datetime: Union[None, str, tuple] = None,
                       freq: Optional[str] = None,
                       groups: Union[None, str, list[str]] = None,
                       add_means=True,
                       add_periods=False,
                       copy=True):
        """
        Add different periodic types and the means

        If 'target' is None, the means is NOT added
        If 'datetime' is not defined, it is used the index. If defined, it must be a valid datetime column
        If 'groups' is defined, the dataframe is split and merged

        :param df: dataframe do process
        :param periodic: flags of periodics to use
        :param target: column name(s) where to compute the means
        :param datetime: datetime column. If not specified, it is used the index
        :param groups: list of colums used to split
        """
        super().__init__(target, copy)
        self.periodic = periodic
        self.freq = freq
        self.datetime = datetime

        if datetime is None or isinstance(datetime, str):
            self.datetime = datetime
        else:
            self.datetime = datetime[0]
            self.freq = datetime[1]

        self.groups = as_list(groups)
        self.add_means = add_means
        self.add_periods = add_periods

        # means for each column and period
        self._means = {}

    def fit(self, X: DataFrame) -> "PeriodicEncoder":
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

    def transform(self, X: DataFrame) -> DataFrame:
        datetime = self.datetime
        targets = self.columns
        periodic = self.periodic
        drop_dt = False

        if not self.add_periods and not self.add_means:
            return X

        if self.copy:
            X = X.copy()

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

        if self.add_means:
            X = _add_means(X, datetime, periodic, targets, self._means)

        if not self.add_periods:
            X = _drop_periods(X, datetime, periodic)

        if drop_dt:
            X.drop(datetime, axis=1, inplace=True)

        return X
# end
