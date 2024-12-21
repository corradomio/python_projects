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

    "split_column",     # DEPERECATED
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
    "columns_rename"
]


import random
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from typing import Union, Optional, Collection, Any
from pandas import CategoricalDtype
from stdlib import NoneType, CollectionType, as_list, as_tuple, is_instance


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

DATAFRAME_OR_DICT = Union[pd.DataFrame, dict[tuple, pd.DataFrame]]
TRAIN_TEST_TYPE = tuple[DATAFRAME_OR_DICT, Union[NoneType, DATAFRAME_OR_DICT]]
PANDAS_TYPE = Union[NoneType, pd.DataFrame, pd.Series, NoneType]


# ---------------------------------------------------------------------------
# safe_sorted
# check_columns
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

def _groups_split_on_columns(df, groups, drop, keep):
    dfdict: dict[tuple, pd.DataFrame] = {}

    # Note: IF len(groups) == 1, Pandas return 'gname' in instead than '(gname,)'
    # The library generates a FutureWarning !!!!
    if len(groups) == 1:
        for g, gdf in df.groupby(by=groups[0]):
            dfdict[(g,)] = gdf
            if keep > 0 and len(dfdict) == keep: break
    else:
        for g, gdf in df.groupby(by=groups):
            dfdict[g] = gdf
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
                  drop=True) -> pd.DataFrame:
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


def set_index(df: pd.DataFrame, columns: Union[None, str, list[str]], *,
              inplace=False,
              drop=False,
              as_datetime: bool = False,
    ) -> pd.DataFrame:
    """
    Create a multiindex based on the columns list

    :param df: dataframe to process
    :param columns: column or list of columns to use in the index
    :param inplace: if to apply the transformation inplace
    :param drop: if to drop the columns
    :return: the new dataframe
    """
    if inplace:
        df.set_index(columns, inplace=inplace, drop=drop)
    else:
        df = df.set_index(columns, inplace=inplace, drop=drop)
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
            dfdict[lv] = dflv
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

def xy_split(*data_list, target: Union[str, list[str]], shared: Union[None, str, list[str]] = None) \
        -> list[PANDAS_TYPE]:
    """
    Split the df in 'data_list' in X, y

    :param data_list: df list
    :param target: target column name
    :param shared: columns shared between 'X' and 'y' (they will be present in both parts)
    :return: list of split dataframes
    """
    target = as_list(target, 'target')
    shared = as_list(shared, 'shared')

    xy_list = []
    for data in data_list:
        assert isinstance(data, pd.DataFrame)

        columns = data.columns.difference(target).union(shared)
        X = data[columns] if len(columns) > 0 else None

        columns = data.columns.difference(columns).union(target).union(shared)
        y = data[columns] if len(columns) > 0 else None

        xy_list += [X, y]
    # end
    return xy_list
# end


def nan_split(*data_list,
              columns: Union[None, str, list[str]] = None,
              ignore: Union[None, str, list[str]] = None,
              empty=True) -> list[PANDAS_TYPE]:
    """
    Remove the columns of the dataframe containing nans

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
    # pandas.NA
    # numpy NaT
    # numpy nan
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

    def _tsize(n) -> int:
        tsize = train_size
        if 0 < test_size < 1:
            tsize = 1 - test_size
        elif test_size > 1:
            tsize = n - test_size
        if 0 < tsize < 1:
            return int(n*tsize)
        elif tsize > 1:
            return tsize
        else:
            return 1

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
        t = _tsize(len(data))

    train = data[:t]
    test_ = data[t:]
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
        pass
    elif c == 1:
        columns = [target]
    else:
        columns = [f'{target}_{i}' for i in range(c)]

    df = pd.DataFrame(data, columns=columns, index=index)
    return df
# end


def to_numpy(data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    """
    Safe version of 'pd.DataFrame.to_numpy()'
    :param data:
    :return:
    """
    if data is None:
        return None
    else:
        return data.to_numpy()
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
