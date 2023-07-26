import logging

import pandas as pd
from numpy import issubdtype, integer, datetime64
from pandas import DataFrame, Period, PeriodIndex

from .base import dataframe_filter_outliers
from .time import infer_freq


# ---------------------------------------------------------------------------
# DataFrameEncoder
# ---------------------------------------------------------------------------

class DataFrameTransformer:
    def __init__(self, col):
        self._col = col
        pass

    def fit(self, X: DataFrame) -> "DataFrameTransformer":
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        return X

    def fit_transform(self, X: DataFrame) -> DataFrame:
        return self.fit(X).transform(X)
# end


# ---------------------------------------------------------------------------
# DatetimeEncoder
# ---------------------------------------------------------------------------

class DatetimeEncoder(DataFrameTransformer):

    def __init__(self, col, format, freq):
        super().__init__(col)
        self._format = format
        self._freq = freq
        self._log = logging.getLogger("DatetimeEncoder")

    def fit(self, X: DataFrame) -> "DatetimeEncoder":
        assert isinstance(X, DataFrame)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)
        if self._col is None:
            return X

        df = X
        df = self._convert_to_datetime(df)
        df = self._copy_to_index(df)
        df = self._infer_freq(df)

        return df

    def _convert_to_datetime(self, df: DataFrame) -> DataFrame:
        col = self._col
        dttype = df[col].dtype.type
        if issubdtype(dttype, datetime64):
            return df
        if dttype in [Period]:
            return df

        # check if the values are strings
        is_str = df[col].apply(type).eq(str).any()

        date_format = self._format
        if date_format is not None and is_str:
            dt = pd.to_datetime(df[col], format=date_format)
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
        col = self._col
        df = df.set_index(df[col])
        # df = df[df.columns.difference([col])]

        # STUPID Pandas if 'df' has index a PeriodIndex, df.to_period() raise an exception!
        if type(df.index) != PeriodIndex:
            df = df.to_period(freq=self._freq)
        return df

    def _infer_freq(self, df: DataFrame) -> DataFrame:
        if df.index.freq is not None:
            return df

        freq = self._freq
        if freq is None:
            freq = infer_freq(df.index)

        if freq is None:
            self._log.warning(f"... unable to infer index 'freq'")
            return df

        n = len(df)
        df = df.asfreq(freq, method='pad')
        if len(df) != n:
            self._log.warning(f"... added {len(df)-n} entries")
        return df
    # end
# end


# ---------------------------------------------------------------------------
# BinaryLabelsEncoder
# ---------------------------------------------------------------------------

class BinaryLabelsEncoder(DataFrameTransformer):
    """
    Convert the values in the column in {0,1}
    """

    def __init__(self, col):
        super().__init__(col)
        self._map = {}

    def fit(self, X: DataFrame, y=None) -> "BinaryLabelsEncoder":
        assert isinstance(X, DataFrame)

        # skip integer columns
        s = X[self._col]
        if issubdtype(s.dtype.type, integer):
            return self

        values = sorted(s.unique())

        # skip if there are 2+ values
        if len(values) > 2:
            pass
        # accept if there is a single value
        elif len(values) == 1:
            v = list(values)[0]
            self._map = {v: 0}
        #
        # Note: sorting the values, the order is exactly
        #       what it is necessary to have!
        #
        #       false < true (in all forms)
        #       off < on
        #       close < open
        #       float values ...
        #
        # elif False in values and True in values:
        #     self._map = {False: 0, True: 1}
        # elif 'False' in values and 'True' in values:
        #     self._map = {'False': 0, 'True': 1}
        # elif 'false' in values and 'true' in values:
        #     self._map = {'false': 0, 'true': 1}
        # elif 'F' in values and 'T' in values:
        #     self._map = {'F': 0, 'T': 1}
        # elif 'f' in values and 't' in values:
        #     self._map = {'f': 0, 't': 1}
        # elif 'off' in values and 'on' in values:
        #     self._map = {'off': 0, 'on': 1}
        # elif 'close' in values and 'open' in values:
        #     self._map = {'close': 0, 'open': 1}
        else:
            self._map = {values[0]: 0, values[1]: 1}
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)

        if len(self._map) > 0:
            col = self._col
            X = X.replace({col: self._map})
        return X
# end


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder(DataFrameTransformer):

    def __init__(self, col):
        super().__init__(col)
        self._map = {}

    @property
    def has_transform(self):
        # for speedup
        return len(self._map) > 0

    def fit(self, X: DataFrame, y=None) -> "OneHotEncoder":
        assert isinstance(X, DataFrame)

        col = self._col
        values = sorted(set(X[col].to_list()))
        n = len(values)
        for i in range(n):
            v = values[i]
            self._map[v] = i

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)
        col = self._col

        # create the columns
        for v in self._map:
            vcol = f"{col}_{v}"
            X[vcol] = 0

        for v in self._map:
            vcol = f"{col}_{v}"
            X.loc[X[col] == v, vcol] = 1

        return X
# end


# ---------------------------------------------------------------------------
# StandardScalerEncoder
# MinMaxEncoder
# MeanStdEncoder
# ---------------------------------------------------------------------------

NO_SCALE_LIMIT = 10
NO_SCALE_EPS = 0.0000001


class StandardScalerEncoder(DataFrameTransformer):

    def __init__(self, col):
        super().__init__(col)
        self._mean = 0.
        self._sdev = 0.

    def fit(self, X: DataFrame, y=None) -> "StandardScalerEncoder":
        assert isinstance(X, DataFrame)
        col = self._col

        values = X[col].to_numpy(dtype=float)
        vmin, vmax = min(values), max(values)

        # if the values are already in a reasonable small range, don't scale
        if -NO_SCALE_LIMIT <= vmin <= vmax <= +NO_SCALE_LIMIT:
            return self
        if (vmax - vmin) <= NO_SCALE_EPS:
            self._mean = values.mean()
        else:
            self._mean = values.mean()
            self._sdev = values.std()
        # end
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)

        if self._sdev <= NO_SCALE_EPS and self._mean == 0.:
            return X

        col = self._col
        values = X[col].to_numpy(dtype=float)
        if self._sdev <= NO_SCALE_EPS:
            values = values - self._mean
        else:
            values = (values - self._mean) / self._sdev

        X[col] = values
        return X

    def inverse_transform(self, X: DataFrame):
        assert isinstance(X, DataFrame)

        if self._sdev <= NO_SCALE_EPS and self._mean == 0.:
            return X

        col = self._col
        values = X[col].to_numpy(dtype=float)
        if self._sdev <= NO_SCALE_EPS:
            values = values + self._mean
        else:
            values = values*self._sdev + self._mean

        X[col] = values
        return X
# end


class MinMaxEncoder(DataFrameTransformer):

    def __init__(self, col):
        super().__init__(col)
        self._min = -float('inf')
        self._delta = +float('inf')

    def fit(self, X: DataFrame, y=None) -> "MinMaxEncoder":
        assert isinstance(X, DataFrame)
        col = self._col

        values = X[col].to_numpy(dtype=float)
        self._min = min(values)
        self._delta = max(values) - self._min

        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)

        col = self._col
        values = X[col].to_numpy(dtype=float)
        values = (values-self._min)/self._delta

        X[col] = values
        return X

    def inverse_transform(self, X: DataFrame):
        assert isinstance(X, DataFrame)

        col = self._col
        values = X[col].to_numpy(dtype=float)
        values = self._min = self._delta*values

        X[col] = values
        return X
# end


class MeanStdEncoder(DataFrameTransformer):

    def __init__(self, col):
        super().__init__(col)
        self._mean = 0.
        self._sdev = 0.

    def fit(self, X: DataFrame, y=None) -> "StandardScalerEncoder":
        assert isinstance(X, DataFrame)
        col = self._col

        values = X[col].to_numpy(dtype=float)
        self._mean = values.mean()
        self._sdev = values.std()
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)

        col = self._col
        values = X[col].to_numpy(dtype=float)
        values = (values - self._mean) / self._sdev

        X[col] = values
        return X

    def inverse_transform(self, X: DataFrame):
        assert isinstance(X, DataFrame)

        col = self._col
        values = X[col].to_numpy(dtype=float)
        values = values*self._sdev + self._mean

        X[col] = values
        return X
# end


# ---------------------------------------------------------------------------
# OutlierTransformer
# ---------------------------------------------------------------------------

class OutlierTransformer(DataFrameTransformer):

    def __init__(self, col, outlier_std=4):
        super().__init__(col)
        self._outlier_std = outlier_std
        self._mean = 0.
        self._sdev = 0.

    def fit(self, X: DataFrame, y=None) -> "OutlierTransformer":
        assert isinstance(X, DataFrame)
        col = self._col
        # values = X[col].to_numpy(dtype=float)
        # self._mean = values.mean()
        # self._sdev = values.std()
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        assert isinstance(X, DataFrame)

        X = dataframe_filter_outliers(X, self._col, self._outlier_std)

        return X
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
