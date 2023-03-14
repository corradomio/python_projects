from abc import ABC
from typing import Union
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

    def __init__(self, cols: Union[str, list[str]], mapping: list=[]):
        if isinstance(cols, str): cols = [cols]
        super().__init__(cols=cols)
        assert isinstance(cols, (list, tuple))
        assert isinstance(mapping, (list, tuple))

        self._mapping: list[str] = mapping

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
            return col + "_" + key.replace(' ', '_').replace('(', '').replace(')', '')

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
# Simple Pipeline
# ---------------------------------------------------------------------------

class Pipeline:

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
# split_partitions
# ---------------------------------------------------------------------------

def partition_lengths(n: int, quota: Union[int, list[int]]) -> list[int]:
    if isinstance(quota, int):
        quota = [1]*quota
    k = len(quota)
    tot = sum(quota)
    lengths = []
    for i in range(k-1):
        l = int(n*quota[i]/tot + 0.6)
        lengths.append(l)
    lengths.append(n - sum(lengths))
    return lengths
# end


def split_partitions(*data_list, partitions: Union[int, list[int]]=1) -> list[list[Union[DataFrame, Series]]]:
    parts_list = []
    for data in data_list:
        n = len(data)
        plengths = partition_lengths(n, partitions)
        pn = len(plengths)
        s=0
        parts = []
        for i in range(pn):
            pl = plengths[i]
            part = data.iloc[s:s+pl]
            parts.append(part)
            s += pl
        # end
        parts_list.append(parts)
    # end
    return parts_list
# end
