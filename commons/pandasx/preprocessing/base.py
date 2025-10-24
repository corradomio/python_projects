from typing import cast

import pandas as pd

from ..base import as_list, NoneType, groups_split, groups_merge, PANDAS_TYPE
from ..base import is_instance


# ---------------------------------------------------------------------------
# BaseEncoder
# ---------------------------------------------------------------------------

class BaseEncoder:

    def __init__(self, columns, copy):
        assert is_instance(columns, (NoneType, str, list[str])), "Invalid 'columns' value type"
        assert is_instance(copy, bool), "Invalid 'copy' value type"
        self.columns = as_list(columns, "columns")
        self.copy = copy
        self._col = self.columns[0] if len(self.columns) > 0 else None
        pass

    def fit(self, X) -> "BaseEncoder":
        assert isinstance(X, pd.DataFrame)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, X) -> pd.DataFrame:
        return self.fit(X).transform(X)

    def _get_columns(self, X) -> str | list[str]:
        # NO: if the list of columns is [], IT REMAINS [] !!!
        # if len(self.columns) > 0:
        #     # return list(X.columns.intersection(self.columns))
        #     return self.columns
        # else:
        #     return list(X.columns)
        if isinstance(X, pd.Series):
            return cast(str, X.name)
        if self.columns is not None:
            return self.columns
        else:
            return list(X.columns)

    def _check_X(self, X: pd.DataFrame):
        assert isinstance(X, (pd.DataFrame, pd.Series))
        return X.copy() if self.copy else X
# end


# ---------------------------------------------------------------------------
# XyBaseEncoder
# ---------------------------------------------------------------------------

class XyBaseEncoder(BaseEncoder):

    def __init__(self, columns, copy):
        super().__init__(columns, copy)

    def fit(self, X, y=None) -> "XyBaseEncoder":
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, (pd.DataFrame, pd.Series))
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, (pd.DataFrame, pd.Series))
        return None

    def fit_transform(self, X, y=None):
        return self.fit(X=X, y=y).transform(X=X, y=y)

    def _check_Xy(self, X, y):
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, (pd.DataFrame, pd.Series))
        return (X.copy(), y.copy()) if self.copy else X, y
# end


# ---------------------------------------------------------------------------
# pd.SeriesBaseEncoder
# ---------------------------------------------------------------------------

class PandasBaseEncoder(BaseEncoder):

    def __init__(self, columns, copy):
        super().__init__(columns, copy)

    def fit(self, X) -> "PandasBaseEncoder":
        assert isinstance(X, (pd.DataFrame, pd.Series))
        if isinstance(X, pd.DataFrame):
            self._fit_df(X)
        else:
            self._fit(X)
        return self

    def _fit_df(self, D: pd.DataFrame):
        columns = self._get_columns(D)
        for col in columns:
            self._fit(D[col])

    def _fit(self, S: pd.Series):
        ...

    def transform(self, X: PANDAS_TYPE) -> PANDAS_TYPE:
        assert isinstance(X, (pd.DataFrame, pd.Series))
        if isinstance(X, pd.DataFrame):
            return self._transform_df(X)
        elif isinstance(X, pd.Series):
            return self._transform(X)

    def _transform_df(self, D: pd.DataFrame) -> pd.DataFrame:
        columns = self._get_columns(D)
        D = self._check_X(D)
        for col in columns:
            D[col] = self._transform(D[col])
        return D

    def _transform(self, S: pd.Series) -> pd.Series:
        ...
    
    def inverse_transform(self, X: PANDAS_TYPE) -> PANDAS_TYPE:
        assert isinstance(X, (pd.DataFrame, pd.Series))
        if isinstance(X, pd.DataFrame):
            return self._inverse_transform_df(X)
        elif isinstance(X, pd.Series):
            return self._inverse_transform(X)
        
    def _inverse_transform_df(self, D: pd.DataFrame) -> pd.DataFrame:
        columns = self._get_columns(D)
        D = self._check_X(D)
        for col in columns:
            D[col] = self._inverse_transform(D[col])
        return D
    
    def _inverse_transform(self, S: pd.Series) -> pd.Series:
        ...
# end


# ---------------------------------------------------------------------------
# SequenceEncoder
# ---------------------------------------------------------------------------

class SequenceEncoder(BaseEncoder):
    """
    Apply a list of encoders in sequence
    """

    def __init__(self, encoders: list[BaseEncoder]):
        super().__init__([], False)
        self.encoders = encoders

    def fit(self, X):
        assert is_instance(X, pd.DataFrame)
        for encoder in self.encoders:
            encoder.fit(X)
        return self

    def transform(self, X):
        assert is_instance(X, pd.DataFrame)
        for encoder in self.encoders:
            X = encoder.transform(X)
        return X

    def inverse_transform(self, X):
        assert is_instance(X, pd.DataFrame)
        for encoder in reversed(self.encoders):
            X = encoder.transform(X)
        return X
# end


# ---------------------------------------------------------------------------
# IdentityEncoder
# ---------------------------------------------------------------------------

class IdentityEncoder(BaseEncoder):
    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


# ---------------------------------------------------------------------------
# GroupsBaseEncoder
# ---------------------------------------------------------------------------
# The existence of the group columns implies the data MUST BE a pd.DataFrame

class GroupsBaseEncoder(BaseEncoder):

    def __init__(self, columns, groups, copy=True):
        super().__init__(columns, copy)
        self.groups = as_list(groups)
        self._params = {}

    def _check_X(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame)
        return X.copy() if self.copy else X

    # -----------------------------------------------------------------------
    # Override
    # -----------------------------------------------------------------------

    def _get_params(self, g):
        return self._params[g]

    def _set_params(self, g, params):
        self._params[g] = params

    def _compute_params(self, g, X):
        ...

    def _apply_transform(self, X, params):
        ...

    def _apply_inverse_transform(self, X, params):
        ...

    # -----------------------------------------------------------------------
    # fit(X)
    # -----------------------------------------------------------------------

    def fit(self, X):
        X = self._check_X(X)

        if len(self.groups) > 0:
            self._fit_by_columns(X)
        elif isinstance(X.index, pd.MultiIndex):
            self._fit_by_index(X)
        else:
            self._fit_plain(X)
        return self

    # -----------------------------------------------------------------------

    def _fit_plain(self, X):
        params = self._compute_params(None, X)
        self._set_params(None, params)
        return self

    def _fit_by_columns(self, X):
        groups = groups_split(X, groups=self.groups, drop=True)
        for g in groups:
            Xg = groups[g]
            params = self._compute_params(g, Xg)
            self._set_params(g, params)
        return self

    def _fit_by_index(self, X):
        groups = groups_split(X, drop=True)
        for g in groups:
            Xg = groups[g]
            params = self._compute_params(g, Xg)
            self._set_params(g, params)
        return self

    # -----------------------------------------------------------------------
    # transform(X)
    # -----------------------------------------------------------------------

    def transform(self, X):
        X = self._check_X(X)

        if len(self.groups) > 0:
            Xt = self._transform_by_columns(X)
        elif isinstance(X.index, pd.MultiIndex):
            Xt = self._transform_by_index(X)
        else:
            Xt = self._transform_plain(X)
        return Xt

    # -----------------------------------------------------------------------

    def _transform_plain(self, X):
        params = self._get_params(None)
        return self._apply_transform(X, params)

    def _transform_by_columns(self, X):
        X_dict: dict = dict()
        groups = groups_split(X, groups=self.groups, drop=True)

        for g in groups:
            Xg = groups[g]
            params = self._get_params(g)
            Xt = self._apply_transform(Xg, params)
            X_dict[g] = Xt
        # end
        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _transform_by_index(self, X):
        X_dict: dict = dict()
        groups = groups_split(X, drop=True)

        for g in groups:
            Xg = groups[g]
            params = self._get_params(g)
            Xt = self._apply_transform(Xg, params)
            X_dict[g] = Xt
        # end
        X = groups_merge(X_dict)
        return X

    # -----------------------------------------------------------------------
    # inverse_transform(X)
    # -----------------------------------------------------------------------

    def inverse_transform(self, X):
        X = self._check_X(X)

        if len(self.groups) > 0:
            Xt = self._inverse_transform_by_columns(X)
        elif isinstance(X.index, pd.MultiIndex):
            Xt = self._inverse_transform_by_index(X)
        else:
            Xt = self._inverse_transform_plain(X)
        return Xt

    # -----------------------------------------------------------------------

    def _inverse_transform_plain(self, X):
        params = self._get_params(None)
        return self._apply_inverse_transform(X, params)

    def _inverse_transform_by_columns(self, X):
        X_dict: dict = dict()
        groups = groups_split(X, groups=self.groups, drop=True)

        for g in groups:
            Xg = groups[g]
            params = self._get_params(g)
            Xt = self._apply_inverse_transform(Xg, params)
            X_dict[g] = Xt
        # end
        X = groups_merge(X_dict, groups=self.groups)
        return X

    def _inverse_transform_by_index(self, X):
        X_dict: dict = dict()
        groups = groups_split(X, drop=True)

        for g in groups:
            Xg = groups[g]
            params = self._get_params(g)
            Xt = self._apply_inverse_transform(Xg, params)
            X_dict[g] = Xt
        # end
        X = groups_merge(X_dict)
        return X
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
