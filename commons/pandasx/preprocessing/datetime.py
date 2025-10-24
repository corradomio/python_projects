import logging
import numpy as np
import pandas as pd

from numpy import issubdtype, datetime64
from pandas import DataFrame, Period, PeriodIndex, to_datetime
from stdlib import is_instance

from .base import BaseEncoder
from ..base import safe_sorted, infer_freq, normalize_freq


def week_of_month(date) -> str:
    first_day_of_month = date.replace(day=1)
    # Calculate the offset to align with a Monday-based week (0 for Monday)
    # Add 1 to make weeks 1-indexed
    return str((date.day - 1 + first_day_of_month.weekday()) // 7 + 1)


# ---------------------------------------------------------------------------
# DatetimeEncoder
# ---------------------------------------------------------------------------

class DateTimeEncoder(BaseEncoder):

    def __init__(self, column, format, freq, index=True, copy=True):
        super().__init__(column, copy)
        assert isinstance(column, str), "Column is not specified"

        self.format = format
        self.freq = freq
        self.index = index
        self._log = logging.getLogger("DatetimeEncoder")

    def transform(self, X: DataFrame) -> DataFrame:
        X = self._check_X(X)

        X = self._convert_to_datetime(X)
        X = self._copy_to_index(X)
        X = self._infer_freq(X)

        return X

    def _convert_to_datetime(self, df: DataFrame) -> DataFrame:
        col = self._col
        dttype = df[col].dtype.type
        if issubdtype(dttype, datetime64):
            return df
        if dttype in [Period]:
            return df

        # check if the values are strings
        is_str = df[col].apply(type).eq(str).any()

        date_format = self.format
        if date_format is not None and is_str:
            dt = to_datetime(df[col], format=date_format)
            df[col] = dt
        return df
    # end

    def _move_to_index(self, df: DataFrame) -> DataFrame:
        col = self._col
        df = df.set_index(df[col])
        df = df[df.columns.difference([col])]

        # STUPID Pandas if 'df' has index a PeriodIndex, df.to_period() raise an exception!
        if type(df.index) != PeriodIndex:
            df = df.to_period()
        return df

    def _copy_to_index(self, df: DataFrame) -> DataFrame:
        if not self.index:
            return df

        col = self._col
        # ONLY if the index is NOT jet a PeriodIndex]
        if isinstance(df.index, PeriodIndex):
            return df

        df.set_index(df[col], inplace=True)
        # df = df[df.columns.difference([col])]

        # STUPID Pandas if 'df' has index a PeriodIndex, df.to_period() raise an exception!
        if type(df.index) != PeriodIndex:
            df = df.to_period(freq=self.freq)
        return df

    def _infer_freq(self, df: DataFrame) -> DataFrame:
        if df.index.freq is not None:
            return df

        freq = self.freq
        if freq is None:
            return df

        n = len(df)
        df = df.asfreq(freq, method='pad')
        if len(df) != n:
            self._log.warning(f"... added {len(df)-n} entries")
        return df
    # end
# end


# ---------------------------------------------------------------------------
# DateTimeNameEncoder
# ---------------------------------------------------------------------------

class DateTimeNameEncoder(BaseEncoder):
    """
    Add a column with the name of the datetime, based on the frequency
    If the column is already present, it does nothing
    """

    def __init__(self, datetime, datetime_name, freq=None, copy=True):
        if is_instance(datetime, list):
            datetime, freq = datetime
        super().__init__([datetime, datetime_name], copy)
        assert isinstance(datetime, str)
        assert isinstance(datetime_name, str)
        self.freq = freq
        self._names = None
        self._unique = None

    def fit(self, X):
        datetime = self.columns[0]
        assert datetime in X.columns, f"Column {datetime} not found"
        freq = self.freq
        if freq is None:
            freq = normalize_freq(infer_freq(X[datetime]))

        assert freq in ['D', 'W', 'M'], f"Unsupported freq: {freq}"

        if freq == 'D':
            values = X[datetime].dt.day_name()
        elif freq == 'W':
            values = X[datetime].apply(week_of_month)
        elif freq == 'M':
            values = X[datetime].dt.month_name()
        else:
            raise ValueError(f"Unsupported freq: {freq}")

        self._values = values
        self._unique = safe_sorted(values.unique())
        return self

    def transform(self, X):
        datetime_name = self.columns[1]
        if datetime_name in X.columns:
            uniques = safe_sorted(X[datetime_name].unique())
        else:
            X[datetime_name] = self._values
            uniques = self._unique

        col = datetime_name

        # prepare the columns
        for v in uniques:
            ohcol = f"{col}_{v}"
            X.insert(len(X.columns), ohcol, 0.)
            X[ohcol] = X[ohcol].astype(float)

        # fill the columns
        for v in uniques:
            ohcol = f"{col}_{v}"
            X.loc[X[col] == v, ohcol] = 1.

        # remove the columns
        X.drop([col], axis=1, inplace=True)

        return X
    # end

# end


# ---------------------------------------------------------------------------
# DateTimeToIndexTransformer
# ---------------------------------------------------------------------------

class DateTimeToIndexTransformer(BaseEncoder):

    def __init__(self, datetime, drop=True, copy=True):
        super().__init__([datetime], copy)
        self.freq = None
        self.drop = drop

    def fit(self, X):
        datetime = self._col
        dtser = X[datetime]
        self.freq = infer_freq(dtser)
        pass

    def transform(self, X: pd.DataFrame):
        datetime = self.columns[0]
        dtser = X[datetime]
        X.set_index(dtser, inplace=True)
        if self.freq is not None and not isinstance(X.index, pd.PeriodIndex):
            di: pd.DatetimeIndex = X.index
            pi: pd.PeriodIndex = di.to_period(self.freq)
            X.set_index(pi, inplace=True)
        if self.drop:
            # X = X[X.columns.difference([datetime])]
            X.drop(labels=[datetime], axis=1, inplace=True)

        return X
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
