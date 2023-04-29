from abc import ABC
from random import shuffle
from typing import Union, Tuple, Optional
from pandas import DataFrame, Series
import category_encoders.utils as util

class _Encoder:

    def fit(self, X: DataFrame, y: Union[None, DataFrame, Series]=None) -> "_Encoder":
        ...

    def transform(self, X: DataFrame) -> DataFrame:
        ...

    def fit_transform(self, X: DataFrame, y: Union[None, DataFrame, Series]=None) -> DataFrame:
        ...


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
            df = df.to_period()
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
# StandardScaler
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


# ---------------------------------------------------------------------------
# Extra Encoders
# ---------------------------------------------------------------------------

class PandasCategoricalEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin, ABC, _Encoder):

    def __init__(self, cols: Union[str, list[str]]):
        if isinstance(cols, str): cols = [cols]
        super().__init__(cols=cols)
        assert isinstance(cols, (list, tuple))

    def fit(self, X: DataFrame, y=None, **kwargs):
        assert isinstance(X, DataFrame)
        return self

    def transform(self, X: DataFrame, override_return_df=False) -> DataFrame:
        for col in self.cols:
            X[col] = X[col].astype('category')
        return X
# end


class OrderedLabelEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin, ABC, _Encoder):

    def __init__(self, cols: Union[str, list[str]], mapping: list=[], remove_chars=None):
        if isinstance(cols, str): cols = [cols]
        super().__init__(cols=cols)
        assert isinstance(cols, (list, tuple))
        assert isinstance(mapping, (list, tuple))

        self._mapping: list[str] = mapping
        self._remove_chars = '()[]{}' if remove_chars is None else remove_chars

    def fit(self, X: DataFrame, y=None, **kwargs):
        assert isinstance(X, DataFrame)
        if len(self._mapping) == 0:
            col = self.cols[0]
            self._mapping = sorted(X[col].unique())
        return self

    def transform(self, X: DataFrame, override_return_df=False) -> DataFrame:
        for col in self.cols:
            if len(self._mapping) <= 2:
                X = self._map_single_column(X, col)
            else:
                X = self._map_multiple_colmns(X, col)
        return X

    def _map_single_column(self, X, col):
        l = self._mapping
        n = len(l)
        mapping = {l[i]: i for i in range(n)}
        X[col] = X[col].replace(mapping)
        return X

    def _map_multiple_colmns(self, X, col: str):

        def ccol_of(key: str):
            if not isinstance(key, str):
                key = str(key)
            if ' ' in key:
                key = key.replace(' ', '_')
            for c in self._remove_chars:
                if c in key:
                    key = key.replace(c, '')
            return col + "_" + key

        for key in self._mapping:
            ccol = ccol_of(key)
            X[ccol] = 0
            X.loc[X[col] == key, ccol] = 1

        X = X[X.columns.difference([col])]
        return X
# end


class DTypeEncoder(util.BaseEncoder, util.UnsupervisedTransformerMixin, ABC, _Encoder):

    def __init__(self, cols: Union[str, list[str]], dtype):
        if isinstance(cols, str): cols = [cols]
        super().__init__(cols=cols)
        self._dtype = dtype

    def fit(self, X: DataFrame, y=None, **kwargs):
        assert isinstance(X, DataFrame)
        return self

    def transform(self, X: DataFrame, override_return_df=False) -> DataFrame:
        for col in self.cols:
            X[col] = X[col].astype(self._dtype)
        return X
# end


class IgnoreTransformer(util.BaseEncoder, util.UnsupervisedTransformerMixin, ABC, _Encoder):

    def __init__(self, cols: Union[str, list[str]]):
        if isinstance(cols, str): cols = [cols]
        super().__init__(cols=cols)

    def fit(self, X: DataFrame, y=None, **kwargs):
        return self

    def transform(self, X: DataFrame, override_return_df=False) -> DataFrame:
        X = X[X.columns.difference(self.cols)]
        return X
# end


# ---------------------------------------------------------------------------
# ShuffleEncoder
# ---------------------------------------------------------------------------

class ShuffleTransformer(util.BaseEncoder, util.UnsupervisedTransformerMixin, ABC, _Encoder):

    def __init__(self, cols: Union[str, list[str]]=[]):
        if isinstance(cols, str): cols = [cols]
        super().__init__(cols=cols)
        self._indices = []

    def fit_transform(self, X: DataFrame, y: Optional[DataFrame]=None):
        return self.fit(X, y).transform(X, y)

    def fit(self, X: DataFrame, y: Optional[DataFrame]=None, **kwargs):
        assert isinstance(X, DataFrame)
        n = len(X)
        self._indices = list(range(n))
        shuffle(self._indices)
        return self

    def transform(self, X: DataFrame, y=None) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
        X = X.iloc[self._indices]
        if y is None:
            return X
        y = y.iloc[self._indices]
        return X, y


# ---------------------------------------------------------------------------
# Simple Pipeline
# ---------------------------------------------------------------------------

class TransformerPipeline:

    def __init__(self, steps: list[_Encoder]):
        self._steps: list[_Encoder] = steps

    def fit(self, X: DataFrame, y=None) -> "Pipeline":
        for step in self._steps:
            X = step.fit_transform(X, y)
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        for step in self._steps:
            X = step.transform(X)
        return X

    def fit_transform(self, X: DataFrame, y: Union[None, DataFrame, Series]=None) -> DataFrame:
        for step in self._steps:
            X = step.fit_transform(X, y)
        return X
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
