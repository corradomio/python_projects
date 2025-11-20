#
# Pandas reader
# Some simple data set loaders: from csv and .arff data files
#

__all__ = [
    # "find_binary",
    # "binary_encode",
    "dataframe_sort",

    "groups_list",
    "groups_split",
    "groups_merge",
    "groups_select",
    "groups_count",
    "groups_apply",
    "groups_set",

    "split_column",     # DEPRECATED
    "merge_column",     # DEPRECATED

    "columns_split",
    "columns_merge",
    "columns_range",

    "multiindex_get_level_values",
    "set_index", "set_multiindex",
    "index_split",
    "index_merge",

    "xy_split",
    "nan_split",
    "nan_drop",
    "nan_set",
    "train_test_split",
    "cutoff_split",

    "type_encode",
    "count_encode",

    "series_argmax",
    "series_argmin",
    "series_range",
    "series_unique_values",

    "dataframe_correlation",
    "filter_outliers",
    "clip_outliers",

    "index_labels",
    "find_unnamed_columns",

    "columns_ignore", "columns_drop",
    "columns_rename",

    "to_numpy",
    "to_dataframe",

    "infer_freq",
    "normalize_freq",
    "FREQUENCIES"
]


import random
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from typing import Union, Optional, Collection, Any, cast
from pandas import CategoricalDtype
from stdlib import NoneType, CollectionType, as_list, as_tuple, is_instance


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

DATAFRAME_OR_DICT = Union[pd.DataFrame, dict[tuple, pd.DataFrame]]
TRAIN_TEST_TYPE = tuple[DATAFRAME_OR_DICT, Union[NoneType, DATAFRAME_OR_DICT]]
PANDAS_TYPE = Union[pd.DataFrame, pd.Series]
OPTIONAL_PANDAS_TYPE = Union[NoneType, pd.DataFrame, pd.Series]


# ---------------------------------------------------------------------------
# safe_sorted
# validate_columns
# ---------------------------------------------------------------------------

def safe_sorted(values):
    ivals = sorted([v for v in values if isinstance(v, int)])
    svals = sorted([v for v in values if isinstance(v, str)])
    uvals = list(set(values).difference(ivals + svals))
    return ivals + svals + uvals


def validate_columns(df: pd.DataFrame, columns: Union[None, str, list[str]]):
    dfcols = df.columns
    columns = as_list(columns)
    invalid = []
    for col in columns:
        if col not in dfcols:
            invalid.append(col)
    if len(invalid) > 0:
        raise ValueError(f"Columns {invalid} not present in DataFrame")


# ---------------------------------------------------------------------------
# dataframe_sort
# ---------------------------------------------------------------------------

def dataframe_sort(df: pd.DataFrame, *, sort: Union[bool, str, list[str]] = True, ascending=True) \
        -> pd.DataFrame:
    if sort in [None, False, [], ()]:
        return df
    if sort is True:
        return df.sort_index(axis=0, ascending=ascending)
    else:
        sort = as_list(sort, 'sort')
        return df.sort_values(axis=0, by=sort, ascending=ascending)
# end


# ---------------------------------------------------------------------------
# groups_list
# groups_count
# ---------------------------------------------------------------------------

def _groups_list_by_columns(df, groups) -> list[tuple]:

    unique_values = df[groups].drop_duplicates(inplace=False, ignore_index=True).values.tolist()
    unique_values = list(map(tuple, unique_values))
    return unique_values
# end


def _groups_list_by_index(df):
    assert isinstance(df.index, pd.MultiIndex)
    index = list(df.index)
    groups = set()
    for idx in index:
        groups.add(idx[0:-1])

    return list(groups)
# end


def groups_list(df: pd.DataFrame, *,
                groups: Union[None, str, list[str]] = None, sort=True) -> list[tuple]:
    """
    Extract from the df the list of groups.
    The groups can be specified as columns in the df or the df itself has a MultiIndex. In this case
    it is used all levels except the last one, containing the datetime

    :param df: DataFrame to split
    :param sort: if to sort the list
    :param groups: list of columns to use during the split. The columns must be categorical or string

    :return list[tuple[str]: the list of tuples
    """
    groups = as_list(groups, 'groups')

    if len(groups) == 0 and not isinstance(df.index, pd.MultiIndex):
        return [tuple()]

    if len(groups) > 0:
        glist = _groups_list_by_columns(df, groups)
    else:
        glist = _groups_list_by_index(df)

    if sort:
        glist = sorted(glist)

    return glist
# end


def groups_count(df: pd.DataFrame, *,
                 groups: Union[None, str, list[str]] = None) -> int:
    glist = groups_list(df, groups=groups, sort=False)
    return len(glist)
# end


def groups_apply(df: pd.DataFrame, fun, *,
                 groups: Union[None, str, list[str]] = None,
                 sortby: Union[None, str, list[str]] = None) -> pd.DataFrame:
    """
    Apply fun to all groups in the dataframe

    :param df: DataFrame to split
    :param fun: function 'f(df: DataFrame, g: tuple) -> DataFrame' called
    :param groups: list of columns to use during the split. The columns must be categorical or string
    :return:
    """

    df_dict = groups_split(df, groups=groups)
    dfres = {}

    g_sorted = sorted(list(df_dict.keys()))
    for g in g_sorted:
        dfg = df_dict[g]
        res = fun(dfg, g)
        # skip no results
        if res is not None:
            dfres[g] = res

    # end
    df = groups_merge(df_dict, groups=groups, sortby=sortby)
    return df
# end


# ---------------------------------------------------------------------------
# groups_split
# ---------------------------------------------------------------------------

def _normalize_tuple(t: tuple):
    # make sure that the tuple is composed by str or int, NOT numpy values
    if len(t) == 0:
        return t
    elif isinstance(t[0], np.integer):
        return tuple(int(e) for e in t)
    else:
        return t


def _groups_split_on_columns(df, groups, drop, keep):
    dfdict: dict[tuple, pd.DataFrame] = {}

    # Note: IF len(groups) == 1, Pandas return 'gname' in instead than '(gname,)'
    # The library generates a FutureWarning !!!!
    if len(groups) == 1:
        for g, gdf in df.groupby(by=groups[0]):
            dfdict[(g,)] = gdf.copy()
            if keep > 0 and len(dfdict) == keep: break
    else:
        for g, gdf in df.groupby(by=groups):
            g = _normalize_tuple(g)
            dfdict[g] = gdf.copy()
            if keep > 0 and len(dfdict) == keep: break

    if drop:
        for g in dfdict:
            gdf = dfdict[g]
            gdf.drop(groups, inplace=True, axis=1)
    # end

    return dfdict
# end


def _groups_split_on_index(df, drop, keep):
    # the multiindex is converted in a plain index
    dfdict = index_split(df, levels=-1, drop=drop)

    if keep > 0:
        for k in reversed(list(dfdict.keys())):
            if len(dfdict) > keep:
                del dfdict[k]

    if not drop:
        # it is necessary to recreate the multiindex
        pass

    return dfdict
# end


def groups_split(df: pd.DataFrame, *,
                 groups: Union[None, str, list[str]] = None,
                 drop=True,
                 keep=0) \
        -> dict[tuple, pd.DataFrame]:
    """
    Split the dataframe based on the content of 'group' columns list or the MultiIndex.

    If 'groups' is None or the empty list AND df has a norma index, it is returned a dictionary
    with key the 'empty tuple' '()'

    If the dataset has as index a MultIndex, it is split based on the first 'n-levels-1'.
    The last level must be a PeriodIndex

    :param df: DataFrame to split
    :param groups: list of columns to use during the split. The columns must be categorical or string.
        If None, it is used the MultiIndex
    :param drop: if to remove the 'groups' columns or from the index
    :param keep:  [DEBUG] keep only the first 'keep' groups

    :return dict[tuple[str], DataFrame]: a dictionary
    """
    assert isinstance(df, pd.DataFrame)
    groups = as_list(groups, 'groups')

    if len(groups) == 0 and not isinstance(df.index, pd.MultiIndex):
        return {tuple(): df}

    if len(groups) > 0:
        dfdict = _groups_split_on_columns(df, groups, drop, keep)
    else:
        dfdict = _groups_split_on_index(df, drop, keep)

    return dfdict
# end


# ---------------------------------------------------------------------------
# groups_merge
# ---------------------------------------------------------------------------

def _groups_merge_by_columns(dfdict, groups, sortby):
    n = len(groups)
    dfonly = []
    for g in dfdict:
        assert len(g) == len(groups)
        gdf = dfdict[g]
        gdf = gdf.copy()

        for i in range(n):
            gdf[groups[i]] = g[i]

        dfonly.append(gdf)
    # end

    df = pd.concat(dfonly, axis=0)
    if len(sortby) > 0:
        df.sort_values(*sortby, inplace=True)

    # put groups columns in first positions
    df = df[list(groups) + list(df.columns.difference(groups))]

    return df
# end


def _groups_merge_by_index(dfdict):
    dfonly = []
    for g in dfdict:
        gdf = dfdict[g]
        gdf = gdf.copy()
        tuples = []
        for ix in gdf.index:
            tuples.append(g + (ix,))

        gix = pd.MultiIndex.from_tuples(tuples)
        gdf.set_index(gix, inplace=True)

        dfonly.append(gdf)
        pass
    df = pd.concat(dfonly, axis=0)
    return df
# end


def groups_merge(dfdict: dict[tuple[str], pd.DataFrame], *,
                 groups: Union[None, str, list[str]] = None,
                 sortby: Union[None, str, list[str]] = None) \
        -> pd.DataFrame:
    """
    Recreate a df based on the content of 'dfdict' and the list of groups.
    Note: the number of columns in 'groups' must be the same of the tuple's length.

    The result dataframe can be reordered base on 'sortby' columns

    :param dfdict: dictionary of dataframes
    :param groups: columns used to save the values of the tuples
    :param sortby: columns to use in the sort
    """
    assert isinstance(dfdict, dict)
    groups = as_list(groups, 'groups')
    sortby = as_list(sortby, 'sortby')

    if len(groups) > 0:
        return _groups_merge_by_columns(dfdict, groups, sortby)
    else:
        return _groups_merge_by_index(dfdict)
# end


# ---------------------------------------------------------------------------
# groups_select
# ---------------------------------------------------------------------------

def _split_dict(d: dict) -> tuple[list, list]:
    keys = []
    vals = []
    for k in d:
        keys.append(k)
        vals.append(d[k])
    return keys, vals


def _group_select_by_columns(df: pd.DataFrame, groups: list[str], values: list[str], drop: bool):
    # groups = as_list(groups)
    # values = as_list(values)

    assert len(groups) == len(values), "groups and values don't have the same number of elements"

    n = len(groups)
    selected_df: pd.DataFrame = df
    for i in range(n):
        g = groups[i]
        v = values[i]
        selected_df = selected_df[selected_df[g] == v]
    # end

    selected_df = selected_df.copy()
    if drop:
        selected_df.drop(groups, inplace=True, axis=1)
    return selected_df
# end


def _groups_select_by_index(df: pd.DataFrame, values: tuple, drop: bool):
    # if 'drop=True', the index levels change!
    # default behavior of 'df.loc[...]'

    if None not in values and drop:
        selected = df.loc[values]
        return selected

    selected = df
    # drop is False or some levels are skipped
    level = 0
    for v in values:
        if v is None:
            level += 1
            continue
        selected = selected.xs(v, level=level, drop_level=drop)
        if not drop:
            level += 1
    # end
    return selected
# end


def groups_select(df: pd.DataFrame,
                  values: Union[None, str, list[str], tuple[str], dict[str, Any]], *,
                  groups: Union[None, str, list[str], tuple[str]] = None,
                  drop=False) -> pd.DataFrame:
    """
    Select the sub-dataframe based on the list of columns & values.
    To use the dataframe index (a multiindex), 'groups' must be None.
    If it is used the dataframe MultiIndex, it is necessary to specify all consecutive levels.
    To skip a level, to use None

    :param df: dataframe to analyze
    :param groups: list of columns or None if it is used the dataframe index (a MultiIndex)
    :param values: list of values (one for each column/index level)
    :param drop: if to delete the columns/levels used in the selection
    :return: the selected dataframe
    """
    assert isinstance(df, pd.DataFrame)

    if isinstance(values, dict):
        groups, values = _split_dict(values)

    groups = as_list(groups)
    values = as_tuple(values)

    if len(groups) == 0:
        return _groups_select_by_index(df, values, drop)
    else:
        return _group_select_by_columns(df, groups, values, drop)
# end


# ---------------------------------------------------------------------------
# groups_set
# ---------------------------------------------------------------------------

def groups_set(df: Union[pd.DataFrame, pd.Series],
               values: Union[None, str, list[str], tuple[str], dict[str, Any]], *,
               groups: Union[None, str, list[str], tuple[str]] = None,
               inplace=False):
    assert isinstance(df, (pd.DataFrame, pd.Series))

    if isinstance(values, dict):
        groups, values = _split_dict(values)

    if isinstance(df, pd.Series):
        df = df.to_frame(cast(pd.Series, df).name)
    elif not inplace:
        df = df.copy()

    for i, col in enumerate(groups):
        df[col] = values[i]

    return df

# ---------------------------------------------------------------------------
# split_column
# merge_column
# ---------------------------------------------------------------------------

def split_column(df: pd.DataFrame, column: str, *,
                 columns: Optional[list[str]] = None,
                 sep: str = '~',
                 drop=False,
                 inplace=False) -> pd.DataFrame:
    """
    Split the content (a string) of the selected 'col', based on the specified separator 'sep'.
    then, it created 2 or more columns with the specified names ('columns') or based on the
    original column name ('col') adding the suffix '_1', '_2', ...

    :param df: dataframe to process
    :param col: column to analyze
    :param columns: columns to generate
    :param sep: separator to use
    :param inplace: if to transform the dataframe inplace
    :return: the updated dataframe
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column, str)
    assert isinstance(columns, (NoneType, list))
    assert isinstance(sep, str)

    data = df[column]
    n = len(data)

    # analyze the first row:
    s = data.iloc[0]
    parts = s.split(sep)
    p = len(parts)

    splits = [[] for p in parts]

    # analyze the data
    for i in range(n):
        s = data.iloc[i]
        parts = s.split(sep)
        for j in range(p):
            splits[j].append(parts[j])
    # end

    # create the columns if not specified
    if columns is None:
        columns = [f'col_{j+1}' for j in range(p)]

    # populate the dataframe
    if not inplace:
        df = df.copy()

    # drop the column if required
    if drop:
        df.drop(column, inplace=True, axis=1)

    for j in range(p):
        df[columns[j]] = splits[j]

    return df
# end


def merge_column(df: pd.DataFrame,
                  columns: list[str], column: str, *,
                  sep: str = '~',
                  drop=False,
                  inplace=False) -> pd.DataFrame:
    """
    Merge the content of 2 or more columns in a single one of type 'string'
    :param df: dataframe to analyze
    :param columns: columns to merge
    :param column: name of the new column
    :param sep: separator used to concatenate the strings
    :param drop: if to remove the merged columns
    :param inplace: if to transform the dataframe inplace
    :return: new dataframe
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(column, str)
    assert isinstance(columns, list)
    assert isinstance(sep, str)

    n = len(df)
    slist: list[pd.Series] = [df[col] for col in columns]
    text: list[str] = []

    for i in range(n):
        tlist = [str(s.iloc[i]) for s in slist]
        text.append(sep.join(tlist))

    if not inplace:
        df = df.copy()
    if drop:
        df = df.drop(columns, axis=1)

    df[column] = text
    return df
# end


# ---------------------------------------------------------------------------
# columns_split
# columns_merge
# ---------------------------------------------------------------------------

def columns_split(df: pd.DataFrame, *,
                  columns: Union[None, str, list[str]] = None,
                  ignore: Union[None, str, list[str]] = None,
                  shared: Union[None, str, list[str]] = None) \
        -> list[pd.DataFrame]:
    """
    Split the dataframe in a list of series based on the list of selected columns
    """
    assert isinstance(df, pd.DataFrame)
    columns = as_list(columns, 'columns')
    ignore = as_list(ignore, 'ignore')
    shared = as_list(shared, 'shared')

    if len(columns) == 0:
        columns = list(df.columns.difference(ignore + shared))

    parts = []
    for col in columns:
        parts.append(df[[col] + shared])

    return parts
# end


def columns_merge(parts: list[pd.DataFrame], *, sort: Union[None, bool, list[str]] = None) -> pd.DataFrame:
    assert is_instance(parts, Collection[pd.DataFrame]) and len(parts) > 0

    dfm = parts[0].copy()
    for df in parts:
        acols = df.columns.difference(dfm.columns)
        if len(acols) == 0:
            continue
        dfm = pd.concat([dfm, df[acols]], axis=1)

    scols = []
    if sort is True:
        scols = sorted(dfm.columns)
    elif is_instance(sort, Collection[str]):
        scols = sort + list(dfm.columns.difference(sort))

    if len(scols) > 0:
        dfm = dfm[scols]
    return dfm


# ---------------------------------------------------------------------------
# columns_range
# ---------------------------------------------------------------------------

class RangeValues:
    def __init__(self, minv, maxv):
        self.minv = minv
        self.maxv = maxv
        self.values = None

    def bounds(self, n=None):
        if n in [None, 0]:
            return self.minv, self.maxv
        else:
            dv = (self.maxv - self.minv) / (n-1)
            self.values = [self.minv + i*dv for i in range(n)]
            return self.values

    def random(self):
        return random.uniform(self.minv, self.maxv) \
            if self.values is None else random.choice(self.values)


class ListValues:
    def __init__(self, values):
        self.values = list(values)

    def bounds(self, n=None):
        return self.values

    def random(self):
        return random.choice(self.values)


def _ndarray_ranges(df: np.ndarray):
    ranges = {}
    dtype = df.dtype
    if len(df.shape) == 1:
        df = df.reshape((-1, 1))
    if dtype in [float, np.float16, np.float32, np.float64]:
        for col in range(df.shape[1]):
            values = df[:, col]
            ranges[col] = RangeValues(values.min(), values.max())
    elif dtype in [int, np.int8, np.int16, np.int32]:
        for col in range(df.shape[1]):
            values = df[:, col]
            ranges[col] = RangeValues(values.min(), values.max())
    elif dtype in [str]:
        for col in range(df.shape[1]):
            values = df[:, col]
            ranges[col] = ListValues(values.unique())
    else:
        raise ValueError(f"Unsupported dtype {dtype}")
    return ranges


def _dataframe_ranges(df: pd.DataFrame, min_counts):
    ranges = {}
    for col in df.columns:
        values = df[col]
        dtype = df[col].dtype
        vtype = type(values[0])
        if dtype in [float, np.float16, np.float32, np.float64]:
            ranges[col] = RangeValues(values.min(), values.max())
        elif isinstance(dtype, CategoricalDtype):
            ranges[col] = ListValues(values.unique())
        elif dtype in [int, np.int8, np.int16, np.int32] and 0 < len(values.unique()) <= min_counts:
            ranges[col] = ListValues(values.unique())
        elif dtype in [int, np.int8, np.int16, np.int32]:
            ranges[col] = RangeValues(values.min(), values.max())
        elif dtype in [str] or vtype in [str]:
            ranges[col] = ListValues(values.unique())
        else:
            raise ValueError(f"Unsupported dtype {dtype} of column {col}")
    return ranges


def columns_range(df: Union[pd.DataFrame, np.ndarray], min_counts=128):
    """
    Retrieve the range values for all columns.
    If a column is of integer type and there are less than 'min_values' distinct values,
    it is considered a 'categorical column'

    :param df: dataframe to analyze
    :param min_values: min number of integer values to consider as categorical
    :return: a dictionary with the range values ofr each column
    """
    if isinstance(df, np.ndarray):
        return _ndarray_ranges(df)
    elif isinstance(df, pd.DataFrame):
        return _dataframe_ranges(df, min_counts)
    else:
        raise ValueError(f"Unsupported df type {type(df)}")


# ---------------------------------------------------------------------------
# multiindex_get_level_values
# set_index
# index_split
# index_merge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# infer_freq
# ---------------------------------------------------------------------------
# Pandas already offer a 'infer_freq' method

FREQUENCIES = {
    # second
    'S': 1,                     # one second
    's': 1,                     # one second

    # minute
    'T': 60,
    'min': 60,                  # one minute

    # hour
    'H': 60 * 60,               # one hour
    'h': 60 * 60,               # one hour
    'BH': 60 * 60,              # business hour
    'CBH': 60 * 60,             # custom business hour

    # day
    'B': 60 * 60 * 24,          # business day (weekday)
    'D': 60 * 60 * 24,          # one absolute day
    'C': 60 * 60 * 24,          # custom business day

    # week
    'W': 60 * 60 * 24 * 7,      # one week, optionally anchored on a day of the week
    'BW': 60 * 60 * 24 * 5,
    'WOM': 60 * 60 * 24 * 7,    # the x-th day of the y-th week of each month
    'LWOM': 60 * 60 * 24 * 7,   # the x-th day of the last week of each month

    # 15 days/half month
    'SM': 60 * 60 * 24 * 15,
    'SME': 60 * 60 * 24 * 15,   # 15th (or other day_of_month) and calendar month end
    'SMS': 60 * 60 * 24 * 15,   # 15th (or other day_of_month) and calendar month begin

    # month
    'M': 60 * 60 * 24 * 30,     # month
    'MS': 60 * 60 * 24 * 30,    # calendar month begin
    'ME': 60 * 60 * 24 * 30,    # calendar month end
    'BM': 60 * 60 * 24 * 30,    # business month
    'BME': 60 * 60 * 24 * 30,   # business month end
    'BMS': 60 * 60 * 24 * 30,   # business month begin
    'CBM': 60 * 60 * 24 * 30,   # custom business month
    'CBME': 60 * 60 * 24 * 30,  # custom business month end
    'CBMS': 60 * 60 * 24 * 30,  # custom business month begin
    'MBS': 60 * 60 * 24 * 30,
    'CMBS': 60 * 60 * 24 * 30,

    # quarter
    'Q': 60 * 60 * 24 * 91,
    'QE': 60 * 60 * 24 * 91,    # calendar quarter end
    'QS': 60 * 60 * 24 * 91,    # calendar quarter start
    'BQ': 60 * 60 * 24 * 91,
    'BQE': 60 * 60 * 24 * 91,   # business quarter end
    'BQS': 60 * 60 * 24 * 91,   # business quarter begin

    # year
    'A': 60 * 60 * 24 * 365,
    'Y': 60 * 60 * 24 * 365,
    'YE': 60 * 60 * 24 * 365,   # calendar year end
    'YS': 60 * 60 * 24 * 365,   # calendar year begin
    'BA': 60 * 60 * 24 * 365,
    'BY': 60 * 60 * 24 * 365,
    'AS': 60 * 60 * 24 * 365,
    'AY': 60 * 60 * 24 * 365,
    'BAE': 60 * 60 * 24 * 365,  # business year end
    'BAS': 60 * 60 * 24 * 365,  # business year begin
    'BYE': 60 * 60 * 24 * 365,  # business year end
    'BYS': 60 * 60 * 24 * 365,  # business year begin

    'REQ': 60 * 60 * 24 * 7 * 52,   # retail (aka 52-53 week) quarter
    'RE': 60 * 60 * 24 * 7 * 52,  # retail (aka 52-53 week) quarter

    # 'L': 1, 'ms': 1,    # milliseconds
    # 'U': 1, 'us': 1,    # microseconds
    # 'N': 1              # nanoseconds

    'WS': 60 * 60 * 24 * 7,  # one week, optionally anchored on a day of the week
    'WE': 60 * 60 * 24 * 7,  # one week, optionally anchored on a day of the week

}


NORMALIZED_FREQ = {
    None: None,
    # second
    'S': 'S',
    's': 'S',

    # minute
    'T': 'min',
    'min': 'min',

    # hour
    'H': 'H',
    'h': 'H',
    'BH': 'H',
    'CBH': 'H',

    # day
    'B': 'D',
    'D': 'D',
    'C': 'D',

    # week
    'W': 'W',       # 7
    'BW': 'W',     # 5
    'WOM': 'W',     # 7
    'LWOM': 'W',    # 7

    # 15 days/half month
    'SM': 'SM',
    'SME': 'SM',
    'SMS': 'SM',

    # month
    'M': 'M',
    'MS': 'M',
    'ME': 'M',
    'BM': 'M',
    'BME': 'M',
    'BMS': 'M',
    'CBM': 'M',
    'CBME': 'M',
    'CBMS': 'M',
    'MBS': 'M',
    'CMBS': 'M',

    # quarter
    'Q': 'Q',
    'QE': 'Q',
    'QS': 'Q',
    'BQ': 'Q',
    'BQE': 'Q',
    'BQS': 'Q',

    # year
    'A': 'Y',
    'Y': 'Y',
    'YE': 'Y',
    'YS': 'Y',
    'BA': 'Y',
    'BY': 'Y',
    'AS': 'Y',
    'AY': 'Y',
    'BAE': 'Y',
    'BAS': 'Y',
    'BYE': 'Y',
    'BYS': 'Y',

    'REQ': 'RE',
    'RE': 'RE',

    'W-MON': 'W',
    'W-TUE': 'W',
    'W-WED': 'W',
    'W-THU': 'W',
    'W-FRI': 'W',
    'W-SAT': 'W',
    'W-SUN': 'W',
}


def normalize_freq(freq):
    return NORMALIZED_FREQ[freq]
    # return freq


def infer_freq(datetime, steps=5, ntries=3, normalize=True) -> str:
    """
    Infer 'freq' checking randomly different positions of the index

    [2024/02/23] implementation simplified using the services offered
                 by pandas.infer_freq
                 It add some extra cases not supported by

    :param datetime: pandas' index to use
    :param steps: number of success results
    :param ntries: maximum number of retries if some check fails
    :return: the inferred frequency
    """
    if isinstance(datetime, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        freq = pd.infer_freq(datetime)
    elif isinstance(datetime, pd.PeriodIndex):
        freq = datetime.iloc[0].freqstr
    # other pandas index types
    elif isinstance(datetime, pd.Index):
        freq = None
    # elif isinstance(datetime, pd.Series) and datetime.dtype == np.dtypes.ObjectDType:
    #     freq = pd.infer_freq(datetime)
    elif isinstance(datetime, pd.Series) and datetime.dtype == pd.PeriodDtype and hasattr(datetime.iloc[0], "freqstr"):
        freq = datetime.iloc[0].freqstr
    elif isinstance(datetime, pd.Period):
        freq = datetime.freqstr
    else:
        freq = pd.infer_freq(datetime)

    if freq is not None:
        return NORMALIZED_FREQ[freq] if normalize else freq

    freq = _infer_freq(datetime, steps, ntries)
    return NORMALIZED_FREQ[freq] if normalize else freq
# end


def _infer_freq(datetime, steps=5, ntries=3):
    n = len(datetime)-steps
    freq = None
    itry = 0
    while itry < ntries:
        i = random.randrange(n)
        tfreq = pd.infer_freq(datetime[i:i+steps])
        if tfreq is None:
            itry += 1
        elif tfreq != freq:
            freq = tfreq
            itry = 0
        else:
            itry += 1
    # end
    return freq
# end


def multiindex_get_level_values(mi: Union[pd.DataFrame, pd.Series, pd.MultiIndex], levels=1) \
        -> list[tuple]:
    """
    Retrieve the multi level values
    :param mi: multiindex or a DataFrame/Series with multiindex
    :param int levels: can be < 0
    :return: a list of tuples
    """
    assert isinstance(mi, (pd.DataFrame, pd.Series, pd.MultiIndex))
    assert isinstance(levels, int)

    if not isinstance(mi, pd.MultiIndex):
        mi = mi.index
        assert isinstance(mi, pd.MultiIndex)

    if levels < 0:
        levels = len(mi.levels) + levels
    assert levels > 0

    n = len(mi)
    values = set()
    for i in range(n):
        lv = mi.values[i][:levels]
        values.add(lv)
    values = list(values)
    return values
# end


def set_index(df: pd.DataFrame,
              columns: Union[str, list[str]], *,
              inplace=False,
              drop=False,
              as_datetime=False,
              freq: Optional[str]=None
    ) -> pd.DataFrame:
    """
    Create a multiindex based on the columns list

    :param df: dataframe to process
    :param columns: column or list of columns to use in the index
    :param inplace: if to apply the transformation inplace
    :param drop: if to drop the columns
    :return: the new dataframe
    """

    assert len(columns) == 1
    col = columns[0]

    if not inplace:
        df = df.copy()

    df.sort_values(by=col, inplace=True)

    ser = df[col]
    if as_datetime:
        ser = pd.to_datetime(ser)
    if freq is not None:
        ser = ser.dt.to_period(freq)

    df.set_index(ser, inplace=True)
    if drop:
        df.drop(columns=columns, inplace=True)

    return df

    if inplace:
        df.set_index(columns, inplace=inplace, drop=drop)
    else:
        df = df.set_index(columns, inplace=inplace, drop=drop)

    # freq = None
    # if isinstance(df.index, pd.DatetimeIndex) and df.index.freq is None:
    #     freq = infer_freq(df.index)
    # if freq is not None:
    #     df.asfreq(freq)

    if as_datetime and isinstance(df.index, pd.PeriodIndex):
        df.set_index(df.index.to_timestamp())
    return df
# end


set_multiindex = set_index


def index_split(df: pd.DataFrame, *, levels: int = -1, drop=True) -> dict[tuple, pd.DataFrame]:
    """
    Split the dataframe based on the first 'levels' values of the multiindex

    :param df: dataframe to process
    :param levels: n of first multiidex levels to consider. Can be < 0
    :param drop: if to drop the levels used during the splitting
    :return: a dictionary
    """
    assert isinstance(df, (pd.DataFrame, pd.Series))
    assert isinstance(df.index, pd.MultiIndex)
    lvalues = multiindex_get_level_values(df.index, levels=levels)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dfdict = {}
        for lv in lvalues:
            dflv = df.loc[lv]
            dfdict[lv] = dflv.copy()
        # end

    if drop:
        return dfdict

    for lv in dfdict:
        dflv: pd.DataFrame = dfdict[lv]
        tuples = []
        for idx in dflv.index:
            tuples.append(lv + (idx,))
        mi = pd.MultiIndex.from_tuples(tuples)
        dflv.set_index(mi, inplace=True)

    return dfdict
# end


def index_merge(dfdict: dict[tuple, pd.DataFrame], names: Union[None, str, list[str]]=None) -> pd.DataFrame:
    """
    Recreate a dataframe using the keys in the dictionary as multiindex

    :param dfdict: a dictionary of dataframes
    :return: the new dataframe
    """
    dflist = []
    for lv in dfdict:
        dflv: pd.DataFrame = dfdict[lv]
        tuples = []
        for idx in dflv.index:
            tuples.append(lv + (idx,))
        mi = pd.MultiIndex.from_tuples(tuples, names=names)
        dflv.set_index(mi, inplace=True)
        dflist.append(dflv)
    # end
    df = pd.concat(dflist, axis=0)
    return df
# end


# ---------------------------------------------------------------------------
# xy_split
# nan_split
# ---------------------------------------------------------------------------

def xy_split(*df_list, target: Union[str, list[str]], shared: Union[None, str, list[str]] = None) \
        -> list[PANDAS_TYPE]:
    """
    Split the df in 'data_list' in X, y

    :param df_list: df list
    :param target: target column name
    :param shared: columns shared between 'X' and 'y' (they will be present in both parts)
    :return: list of split dataframes
    """
    tlist = as_list(target, 'target')
    shared = as_list(shared, 'shared')

    xy_list = []
    for df in df_list:
        assert isinstance(df, pd.DataFrame)

        columns = df.columns.difference(tlist).union(shared)
        X = df[columns] if len(columns) > 0 else None

        if len(shared) == 0 and isinstance(target, str):
            y = df[target]
        else:
            columns = df.columns.difference(columns).union(tlist).union(shared)
            y = df[columns] if len(columns) > 0 else None

        xy_list += [X, y]
    # end
    return xy_list
# end


def nan_split(*data_list,
              columns: Union[None, str, list[str]] = None,
              ignore: Union[None, str, list[str]] = None,
              empty=False) -> list[PANDAS_TYPE]:
    """
    Split the dataframe horizontally based on specified list of columns having nan values

    :param data_list: list of dataframes to analyze
    :param ignore: list of columns to ignore from the analysis (alternative to 'columns')
    :param columns: list of columns to analyze (alternative to 'ignore')
    :param empty: return an empty dataset (True) or None
    :return:
    """

    ignore = as_list(ignore, 'ignore')
    columns = as_list(columns, 'columns')

    vi_list = []
    for data in data_list:
        if len(ignore) > 0:
            columns = data.columns.difference(ignore)
        if len(columns) == 0:
            columns = data.columns

        invalid_rows = data[columns].isnull().any(axis=1)
        invalid = data.loc[invalid_rows]
        valid = data.loc[data.index.difference(invalid.index)]

        if not empty:
            invalid = invalid if len(invalid) > 0 else None
            valid = valid if len(valid) > 0 else None

        vi_list += [valid, invalid]
    # end
    return vi_list
# end


def nan_drop(df: PANDAS_TYPE, *, columns: Union[None, bool, str, list[str]] = None, inplace=False) -> PANDAS_TYPE:
    """
    Drop the rows having NA in the specifided list of columns
    :param df: dataframe to analyze
    :param columns: columns to consider. Possible values
            - None, False: no column is considered
            - True: all columns are considered
            - str: colum's name
            - list[str]: list of columns
    :param inplace:
    :return: the dataframe without the removed rows
    """
    if df is None:
        return None

    if inplace:
        df.dropna(how='any',axis=0, inplace=True)
    else:
        df = df.dropna(how='any',axis=0, inplace=False)

    return df
# end


def nan_fill(df: PANDAS_TYPE, *, fillna=None) -> PANDAS_TYPE:
    # pandas.NA     Not Available?
    # numpy NaT     Not a Time
    # numpy nan     not a number
    # None
    # if fillna is None:
    #     df.fillna(None, inplace=True)
    if fillna in ["na", "NA"]:
        df.fillna(pd.NA, inplace=True)
    elif fillna in ["NaN", "nan", "NAN"]:
        df.fillna(np.nan, inplace=True)
    # elif fillna in ["None", "null"]:
    #     df.fillna(None, inplace=True)
    else:
        df.fillna(fillna, inplace=True)
    return df
# end


# def nan_drop_old(df: PANDAS_TYPE, *, columns: Union[None, bool, str, list[str]] = None, inplace=False) -> PANDAS_TYPE:
#     """
#     Drop the rows having NA in the specifided list of columns
#     :param df: dataframe to analyze
#     :param columns: columns to consider. Possible values
#             - None, False: no column is considered
#             - True: all columns are considered
#             - str: colum's name
#             - list[str]: list of columns
#     :param inplace:
#     :return: the dataframe without the removed rows
#     """
#     if isinstance(columns, str):
#         columns = [columns]
#
#     if df is None:
#         pass
#     elif columns in [None, False]:
#         pass
#     elif columns is True:
#             df.dropna(how='any', axis=0, inplace=True)
#             # if inplace:
#             #     df.dropna(how='any', axis=0, inplace=True)
#             # else:
#             #     df = df.dropna(how='any', axis=0, inplace=False)
#         else:
#             nan_rows = df[columns].isna().any(axis=1)
#             df.drop(nan_rows, axis=0, inplace=True)
#             # if inplace:
#             #     nan_rows = df[columns].isna().any(axis=1)
#             #     df.drop(nan_rows, axis=0, inplace=True)
#             # else:
#             #     valid_rows = df[columns].notna().all(axis=1)
#             #     df = df[valid_rows]
#     return df
# # end


def nan_set(df: PANDAS_TYPE, columns: Union[str, list[str]], *, on: str, ge: Any, inplace=False) -> PANDAS_TYPE:
    """
    Set a list of columns and a list of records to NaN

    :param df: dataframe to preocess
    :param columns: column or columns to use to udpate
    :param on: columns where to do the selection
    :param ge: (great or equal) value used in condition 'df[<on>] >= <ge>'
    :param inplace: if to modify the df in place
    :return: the dataframe updated
    """
    columns = as_list(columns)
    if not inplace:
        df = df.copy()

    df.loc[df[on] >= ge, columns] = pd.NA
    return df


# ---------------------------------------------------------------------------
# cutoff_split
# ---------------------------------------------------------------------------

def type_encode(df: pd.DataFrame, type: Union[str, type], columns: Union[None, str, list[str]]) -> pd.DataFrame:
    columns = as_list(columns, 'columns')
    for col in columns:
        if col not in df.columns: continue
        df[col] = df[col].astype(type)
    return df


def count_encode(df: pd.DataFrame, count: bool) -> pd.DataFrame:
    if count and 'count' not in df.columns:
        df['count'] = 1.
    return df


# ---------------------------------------------------------------------------
# cutoff_split
# ---------------------------------------------------------------------------

def _cutoff_split(df, cutoff, datetime):
    if isinstance(cutoff, pd.PeriodIndex):
        cutoff = cutoff[0]
    if datetime is not None:
        past = df[df[datetime] <= cutoff]
        future = df[df[datetime] > cutoff]
    else:
        past = df[df.index <= cutoff]
        future = df[df.index > cutoff]

    return past, future


def cutoff_split(*data_list, cutoff,
                 datetime: Optional[str] = None,
                 empty=True,
                 groups=None) -> list[PANDAS_TYPE]:

    groups = as_list(groups, 'groups')
    cl = []
    for data in data_list:
        if len(groups) > 0 or isinstance(data.index, pd.MultiIndex):
            df_dict = groups_split(data, groups=groups)
            past_dict = {}
            future_dict = {}
            for key in df_dict:
                df = df_dict[key]
                past, future = _cutoff_split(df, cutoff, datetime)

                past_dict[key] = past
                future_dict[key] = future
            # end
            past = groups_merge(past_dict, groups=groups)
            future = groups_merge(future_dict, groups=groups)
        else:
            past, future = _cutoff_split(data, cutoff, datetime)

        if not empty:
            past = past if len(past) > 0 else None
            future = future if len(future) > 0 else None

        cl += [past, future]
    return cl
# end


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------

def _train_test_split_single(data, train_size, test_size, datetime) \
        -> Union[tuple[pd.DataFrame, pd.DataFrame], tuple[NoneType, NoneType]]:

    def _train_size(n) -> int:
        if 0 < train_size < 1:
            return int(n*train_size)
        elif train_size >= 1:
            return train_size
        if 0 < test_size < 1:
            return int(n*(1-test_size))
        elif test_size >= 1:
            return n - test_size
        else:
            return n

    if data is None:
        return None, None

    if datetime is not None:
        if isinstance(data.index, pd.PeriodIndex):
            end_date = pd.Period(datetime, freq=data.index.freq)
        elif isinstance(data.index, pd.DatetimeIndex):
            end_date = datetime
        else:
            raise ValueError(f"Unsupported index type {type(data.index)}")
        t = len(data[data.index < end_date])
    else:
        t = _train_size(len(data))

    train = data.iloc[:t]
    test_ = data.iloc[t:]
    return train, test_
# end


def _train_test_split_multiindex(data, train_size, test_size, datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    d_data = index_split(data, levels=-1)
    d_trn = {}
    d_tst = {}
    for lv in d_data:
        dlv = d_data[lv]
        trnlv, tstlv = _train_test_split_single(dlv,
                                                train_size=train_size,
                                                test_size=test_size,
                                                datetime=datetime)
        d_trn[lv] = trnlv
        d_tst[lv] = tstlv
    # end
    trn = index_merge(d_trn)
    tst = index_merge(d_tst)

    return trn, tst
# end


def _sort_by(data, sortby):
    if len(sortby) > 0:
        data.sort_values(by=sortby, axis=0, ascending=True, inplace=True)
    return data
# end


def train_test_split(
    *data_list,
    train_size=0, test_size=0,
    datetime: Optional[dt.datetime] = None,
    groups: Union[None, str, list[str]] = None,
    sortby: Union[None, str, list[str]] = None) -> list[PANDAS_TYPE]:
    """
    Split the df in train/test
    If df has a MultiIndex, it is split each sub-dataframe based on the first [n-1]
    levels
    
    It is possible to specify 'train_size' or 'test_size', not both!
    The value of 'train_size'/'test_size' can be a number in range [0,1] or greater than 1

    It is possible to specify a timestamp. In this case, the the train start ad the speciefied
    timestamp
    
    :param data_list: list of dataframes to split
    :param train_size: train size in ratio [0,1] or in number of samples.
    :param test_size: test size in ratio [0,1] or in number of samples.
    :param datetime: datetime to use for the split. If specified, the test start at this
        timestamp
    :return: 
    """
    assert train_size > 0 or test_size > 0 or isinstance(datetime, dt.datetime), "train_size or test_size must be > 0"

    groups = as_list(groups, "groups")
    sortby = as_list(sortby, "sortby")

    tt_list = []

    for data in data_list:
        #
        # It seems to be not a good idea to try to split a None object!
        # This because some other algorithms need X AND ITS index to work
        # correctly.
        # This means that X can be an EMPTY dataframe BUT with a correct index!
        #
        # assert data is not None

        if len(groups) > 0:
            df_dict = groups_split(data, groups=groups)
            d_trn = {}
            d_tst = {}

            for g in df_dict:
                dfg = df_dict[g]

                dfg = _sort_by(dfg, sortby=sortby)
                trn, tst = _train_test_split_single(dfg,
                                                    train_size=train_size,
                                                    test_size=test_size,
                                                    datetime=datetime)
                d_trn[g] = trn
                d_tst[g] = tst
            # end
            trn = groups_merge(d_trn, groups=groups)
            tst = groups_merge(d_tst, groups=groups)
        elif isinstance(data, pd.DataFrame) and isinstance(data.index, pd.MultiIndex):
            trn, tst = _train_test_split_multiindex(data,
                                                    train_size=train_size,
                                                    test_size=test_size,
                                                    datetime=datetime)
        else:
            trn, tst = _train_test_split_single(data,
                                                train_size=train_size,
                                                test_size=test_size,
                                                datetime=datetime)
        tt_list.append(trn)
        tt_list.append(tst)
    # end
    return tt_list
# end


# ---------------------------------------------------------------------------
# to_dataframe
# to_numpy
# ---------------------------------------------------------------------------

def to_dataframe(data: np.ndarray, *, target: Union[str, list[str]], index=None) -> pd.DataFrame:
    """
    Convert a numpy array in a dataframe

    :param data: numpy array
    :param target:
    :param index:
    :return:
    """
    assert isinstance(data, np.ndarray)
    assert isinstance(target, (str, list))

    n, c = data.shape
    if isinstance(target, CollectionType):
        columns = target
    elif c == 1:
        columns = [target]
    else:
        columns = [f'{target}_{i}' for i in range(c)]

    df = pd.DataFrame(data, columns=columns, index=index)
    return df
# end


# def to_numpy(data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
#     """
#     Safe version of 'pd.DataFrame.to_numpy()'
#     :param data:
#     :return:
#     """
#     return None if data is None else data.to_numpy()
# # end


def to_numpy(data: Union[NoneType, pd.Series, pd.DataFrame, np.ndarray], *,
             dtype=None,
             matrix=False) -> Optional[np.ndarray]:
    assert isinstance(data, (NoneType, pd.Series, pd.DataFrame, np.ndarray))

    if data is None:
        return None
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()

    if matrix and len(data.shape) == 1:
        data = data.reshape((-1, 1))

    if dtype is not None and data.dtype != dtype:
        data = data.astype(dtype)
    return data
# end

# ---------------------------------------------------------------------------
# series_argmax
# series_argmin
# series_unique_values
# series_range
# ---------------------------------------------------------------------------

def series_argmax(df: pd.DataFrame, col: Union[str, int], argmax_col: Union[str, int]) -> float:
    """
    Let df a dataframe, search the row in 'argmax_col' with the highest value
    then extract from 'col' the related value

    :param df: database
    :param col: columns where to extract the value
    :param argmax_col: column where to search the maximum value
    :return:
    """
    s = df[argmax_col]
    at = s.argmax()
    key = s.index[at]
    val = df[col][key]
    return val
# end


def series_argmin(df: pd.DataFrame, col: Union[str, int], argmin_col: Union[str, int]) -> float:
    """
    Let df a dataframe, search the row in 'argmin_col' with the lowest value
    then extract from 'col' the related value

    :param df: database
    :param col: columns where to extract the value
    :param argmin_col: column where to search the minimum value
    :return:
    """
    s = df[argmin_col]
    at = s.argmin()
    key = s.index[at]
    val = df[col][key]
    return val
# end


def series_unique_values(df: pd.DataFrame, col: Union[str, int]) -> np.ndarray:
    """
    Retrieve the unique values in the column

    :param df: dataframe
    :param col: colum where to extract the values
    :return: a ndarray with values ordered
    """
    ser: pd.Series = df[col]
    ser: np.ndarray = ser.unique()
    ser.sort()
    return ser
# end


def series_range(df: pd.DataFrame, col: Union[str, int], *, dx: float = 0) -> tuple:
    """
    Retrieve the values range in the column

    :param df: dataframe
    :param col: colum where to extract the values
    :return: tuple with min & max value
    """
    ser: pd.Series = df[col]
    smin = ser.min() - dx
    smax = ser.max() + dx
    return smin, smax
# end


# ---------------------------------------------------------------------------
# dataframe_correlation
# filter_outliers
# clip_outliers
# ---------------------------------------------------------------------------

def dataframe_correlation(df: Union[DATAFRAME_OR_DICT],
                          columns: Union[str, list[str]], *,
                          target: Optional[str] = None,
                          groups: Union[None, str, list[str]] = None) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]

    if target is None:
        target = columns[0]
    else:
        columns = [target] + columns

    if groups is not None:
        dfdict = groups_split(df, groups=groups)
        corr_dict = {}
        for key in dfdict:
            df = dfdict[key]
            dfcorr = dataframe_correlation(df, columns=columns)
            corr_dict[key] = dfcorr
        dfcorr = groups_merge(corr_dict, groups=groups)
        dfcorr.set_index(dfcorr[groups], inplace=True)
        return dfcorr
    # end

    dftc = df[columns]
    corr = dftc.corr().loc[target]

    dfcorr = pd.DataFrame(columns=corr.index)
    dfcorr = dfcorr.append(corr, ignore_index=True)
    return dfcorr
# end


def filter_outliers(df: pd.DataFrame, col: str, outlier_std: float) -> pd.DataFrame:
    if outlier_std <= 0:
        return df

    values = df[col].to_numpy()

    mean = np.mean(values, axis=0)
    sdev = np.std(values, axis=0)
    max_value = mean + (outlier_std * sdev)
    min_value = mean - (outlier_std * sdev)
    median = np.median(values, axis=0)

    values[(values <= min_value) | (values >= max_value)] = median

    df[col] = values
    return df
# end


def clip_outliers(df: Union[DATAFRAME_OR_DICT],
                  columns: Union[str, list[str]], *,
                  outlier_std: float = 3,
                  groups: Union[None, str, list[str]] = None) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]

    if groups is not None:
        dfdict = groups_split(df, groups=groups)
        outl_dict = {}
        for key in dfdict:
            df = dfdict[key]
            dfoutl = clip_outliers(df, columns=columns)
            outl_dict[key] = dfoutl
        dfoutl = groups_merge(outl_dict, groups=groups)
        return dfoutl
    # end

    for col in columns:
        # data = df[col].copy()
        # mean = data.mean()
        # std = data.std()
        # median = data.median()
        #
        # min = mean-outlier_std*std
        # max = mean+outlier_std*std
        #
        # data[data<min] = median
        # data[data>max] = median
        # df[col] = data

        df = filter_outliers(df, col, outlier_std)
    # end
    return df


# ---------------------------------------------------------------------------
# index_labels
# ---------------------------------------------------------------------------

def index_labels(data: Union[pd.DataFrame, pd.Series], *, n_labels: int = -1) -> list[str]:
    """
    Retrieve the first 'n_labels' labels from 'data.index' and replace the rest with
    the empty string

    :param data: dataframe or series with index
    :param n_labels: n of labels to keep
    :return: list of labels + empty strings
    """
    labels = list(data.index)
    if n_labels > 0 and n_labels < len(labels):
        n = len(labels)
        labels = labels[0:n_labels] + ["" for i in range(n - n_labels)]
    return labels
# end


# ---------------------------------------------------------------------------
# find_unnamed_columns
# columns_ignore
# ---------------------------------------------------------------------------

def find_unnamed_columns(df: pd.DataFrame) -> list[str]:
    """
    List of columns with name 'Unnamed: nn'
    :param df: dataframe to analyze
    :return: list of columns (can be empty)
    """
    return [
        col
        for col in df.columns
        if col.startswith('Unnamed:')
    ]
# end


def columns_ignore(df: pd.DataFrame, ignore: Union[str, list[str]]) -> pd.DataFrame:
    """
    Remove a column or list of columns
    :param df: dataframe to analyze
    :param ignore: columns to ignore
    :return: dataframe processed
    """
    ignore = as_list(ignore, "ignore")
    if len(ignore) > 0:
        df = df[df.columns.difference(ignore)]
    return df
# end


# Alias
columns_drop = columns_ignore


# ---------------------------------------------------------------------------
# columns_rename
# ---------------------------------------------------------------------------

def _normalize_rename(columns, rename):
    # clone 'rename' to obtain a local copy
    rename = {} | rename

    # convert list & tuple into a dictionary
    if isinstance(rename, (list, tuple)):
        n = len(rename)
        rename = {
            i: rename[i]
            for i in range(n)
        }

    assert isinstance(rename, dict), "Parameter 'rename' is not a list or a dictionary"

    # convert positional columns to column name
    keys = list(rename.keys())
    for old_c in keys:
        if isinstance(old_c, int):
            idx_c = old_c
            old_c = columns[old_c]
            rename[old_c] = rename[idx_c]
            del rename[idx_c]

        if old_c not in columns:
            del rename[old_c]
    # end

    # it is NOT possible to rename a column with a name already present
    for old_c in rename.keys():
        new_c = rename[old_c]
        if new_c in columns:
            raise ValueError(f"Unable to rename '{old_c}' into '{new_c}' because already present as column")

    return rename
# end


def columns_rename(df: pd.DataFrame, rename: Union[NoneType, list, dict], ignore_invalid=True) -> pd.DataFrame:
    if rename is None or len(rename) == 0:
        return df

    df_columns = df.columns
    rename = _normalize_rename(df_columns, rename)

    df.rename(columns=rename, inplace=True)
    return df
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
