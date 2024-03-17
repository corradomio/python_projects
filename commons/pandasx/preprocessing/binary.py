from numpy import issubdtype, integer
from pandas import DataFrame

from .base import BaseEncoder

# ---------------------------------------------------------------------------
# find_binary
# binary_encode
# ---------------------------------------------------------------------------

# def find_binary(df: pd.DataFrame, columns: Optional[list[str]] = None) -> list[str]:
#     """
#     Select the columns in 'columns' that can be considered 'binary'
#
#     :param df: dataframe
#     :param columns: columns to analyze. If None, all columns
#     :return: list of binary columns
#     """
#     if columns is None:
#         columns = df.columns
#
#     binary_columns = []
#     for col in columns:
#         nuv = len(df[col].unique())
#         if nuv <= 2:
#             binary_columns.append(col)
#     return binary_columns
# # end


# def binary_encode(df: pd.DataFrame, columns: Union[str, list[str]] = None) -> pd.DataFrame:
#     """
#     Encode the columns values as {0,1}, if not already encoded.
#     It is possible to encode only 1 or 2 distinct values.
#     The values are ordered
#
#     :param df: dataframe
#     :param columns: columns to convert
#     :return: the dataframe with the encoded columns
#     """
#     assert isinstance(df, pd.DataFrame)
#     columns = as_list(columns, 'columns')
#
#     for col in columns:
#         s = df[col]
#         if issubdtype(s.dtype.type, integer):
#             continue
#
#         values = sorted(s.unique())
#         assert len(values) <= 2
#
#         if len(values) == 1 and values[0] in [0, 1]:
#             continue
#         elif values[0] in [0, 1] and values[1] in [0, 1]:
#             continue
#         elif len(values) == 1:
#             v = list(values)[0]
#             map = {v: 0}
#         else:
#             map = {values[0]: 0, values[1]: 1}
#
#         s = s.replace({col: map})
#         df[col] = s
#     # end
#     return df
# # end

# ---------------------------------------------------------------------------
# BinaryLabelsEncoder
# ---------------------------------------------------------------------------

# class BinaryLabelsEncoder(BaseEncoder):
#     """
#     Convert not integer values in the column in {0,1}
#     It is applied to columns with 1 or 2 not integer values ONLY
#     Note: OneHotEncoder is able to encode binary columns in the correct way
#     """
#
#     def __init__(self, columns=None, copy=True):
#         super().__init__(columns, copy)
#         self._maps = {}
#
#     def fit(self, X: DataFrame) -> "BinaryLabelsEncoder":
#         X = self._check_X(X)
#
#         for col in self._get_columns(X):
#             x = X[col]
#             if issubdtype(x.dtype.type, integer):
#                 continue
#
#             values = sorted(x.unique())
#             nv = len(values)
#
#             if nv == 1 and values[0] in [0, 1]:     # skip if a single value in [0, 1]
#                 continue
#             if values == [0, 1]:                    # skip if values in [0, 1]
#                 continue
#             if nv > 2:                              # skip if there are 3+ values
#                 continue
#
#             if nv == 1:                             # v -> 0
#                 v = values[0]
#                 vmap = {v: 0}
#             else:                                   # u -> 0, v -> 1
#                 u, v = values
#                 vmap = {u: 0, v: 1}
#
#             self._maps[col] = vmap
#         # end
#         return self
#
#     def transform(self, X: DataFrame) -> DataFrame:
#         X = self._check_X(X)
#
#         X.replace(self._maps, inplace=True)
#
#         # for col in self._get_columns(X):
#         #     if col not in self._maps:
#         #         continue
#         #
#         #     vmap = self._maps[col]
#         #     X.replace({col: vmap}, inplace=True)
#         # # end
#         return X
# # end
