import logging

from numpy import issubdtype, datetime64
from pandas import DataFrame, Period, PeriodIndex, to_datetime

from .base import BaseEncoder


# ---------------------------------------------------------------------------
# DatetimeEncoder
# ---------------------------------------------------------------------------

class DatetimeEncoder(BaseEncoder):

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
# End
# ---------------------------------------------------------------------------
