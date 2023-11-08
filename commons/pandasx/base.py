#
# Pandas reader
# Some simple data set loaders: from csv and .arff data files
#
import warnings
from typing import List, Union, Optional
from stdlib import NoneType, CollectionType, as_list

import numpy as np
import pandas as pd
from numpy import issubdtype, integer


__all__ = [
    "find_binary",
    "binary_encode",
    "onehot_encode",

    "datetime_encode",
    "datetime_reindex",

    "dataframe_sort",

    "groups_list",
    "groups_split",
    "groups_merge",
    "groups_select",

    "split_column",
    "columns_split",
    "columns_merge",

    "multiindex_get_level_values",
    "set_index",
    "index_split",
    "index_merge",

    "xy_split",
    "nan_split",
    "train_test_split",
    "cutoff_split",

    "series_argmax",
    "series_argmin",
    "series_range",
    "series_unique_values",

    "dataframe_correlation",
    "filter_outliers",
    "clip_outliers",

    "index_labels",
    "find_unnamed_columns",

    "ignore_columns"
]


# ---------------------------------------------------------------------------
# find_binary
# onehot_encode
# datetime_encode
# datetime_reindex
# ---------------------------------------------------------------------------

def find_binary(df: pd.DataFrame, columns: Optional[list[str]] = None) -> list[str]:
    """
    Select the columns in 'columns' that can be considered 'binary'

    :param df: dataframe
    :param columns: columns to analyze. If None, all columns
    :return: list of binary columns
    """
    if columns is None:
        columns = df.columns

    binary_columns = []
    for col in columns:
        nuv = len(df[col].unique())
        if nuv <= 2:
            binary_columns.append(col)
    return binary_columns
# end


def binary_encode(df: pd.DataFrame, columns: Union[str, list[str]] = None) -> pd.DataFrame:
    """
    Encode the columns values as {0,1}, if not already encoded.
    It is possible to encode only 1 or 2 distinct values.
    The values are ordered

    :param df: dataframe
    :param columns: columns to convert
    :return: the dataframe with the encoded columns
    """
    assert isinstance(df, pd.DataFrame)
    columns = as_list(columns, 'columns')

    for col in columns:
        s = df[col]
        if issubdtype(s.dtype.type, integer):
            continue

        values = sorted(s.unique())
        assert len(values) <= 2

        if len(values) == 1 and values[0] in [0, 1]:
            continue
        elif values[0] in [0, 1] and values[1] in [0, 1]:
            continue
        elif len(values) == 1:
            v = list(values)[0]
            map = {v: 0}
        else:
            map = {values[0]: 0, values[1]: 1}

        s = s.replace({col: map})
        df[col] = s
    # end
    return df
# end


def onehot_encode(df: pd.DataFrame, columns: Union[str, list[str]]) -> pd.DataFrame:
    """
    Add some columns based on pandas' 'One-Hot encoding' (pd.get_dummies)

    :param pd.DataFrame df:
    :param list[str] columns: list of columns to convert using 'pd.get_dummies'
    :return pd.DataFrame: new dataframe
    """
    assert isinstance(df, pd.DataFrame)
    columns = as_list(columns, 'columns')

    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col).astype(int)
        df = df.join(dummies)
    return df
# end


def datetime_encode(df: pd.DataFrame,
                    datetime: Union[str, tuple[str]],
                    format: Optional[str] = None,
                    freq: Optional[str] = None) -> pd.DataFrame:
    """
    Convert a string column in datatime/period, based on pandas' 'to_datetime' (pd.to_datetime)

    :param df: dataframe to process
    :param datetime: col | (col, format) | (col, format, freq)
    :param format: datetime format
    :param freq: period frequency
    :return: the df with the 'datetime' column converted
    """
    assert isinstance(datetime, (str, list, tuple))
    assert isinstance(format, (NoneType, str))
    assert isinstance(freq, (NoneType, str))
    # assert 1 < len(datetime) < 4
    if isinstance(datetime, str):
        pass
    elif len(datetime) == 1:
        pass
    elif len(datetime) == 2:
        datetime, format = datetime
    else:
        datetime, format, freq = datetime

    if format is not None:
        df[datetime] = pd.to_datetime(df[datetime], format=format)
    if freq is not None:
        df[datetime] = df[datetime].dt.to_period(freq)
    return df
# end


def datetime_reindex(df: pd.DataFrame, keep='first', mehod='pad') -> pd.DataFrame:
    """
    Make sure that the datetime index in dataframe is complete, based
    on the index's 'frequency'
    :param df: dataframe to process
    :param keep: used in 'index.duplicated(leep=...)'
    :param method: used in 'index.reindex(method=...)'
    :return: reindexed dataframe
    """
    start = df.index[0]
    dtend = df.index[-1]
    freq = start.freq
    dtrange = pd.period_range(start, dtend+1, freq=freq)
    # remove duplicated index keys
    df = df[~df.index.duplicated(keep=keep)]
    # make sure that all timestamps are present
    df = df.reindex(index=dtrange, method=mehod)
    return df
# end


# ---------------------------------------------------------------------------
# dataframe_sort
# ---------------------------------------------------------------------------

def dataframe_sort(df: pd.DataFrame, sort: Union[bool, str, list[str]] = True, ascending=True) \
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
# groups_split
# groups_merge
# ---------------------------------------------------------------------------

DATAFRAME_OR_DICT = Union[pd.DataFrame, dict[tuple, pd.DataFrame]]
TRAIN_TEST_TYPE = tuple[DATAFRAME_OR_DICT, Union[NoneType, DATAFRAME_OR_DICT]]
PANDAS_TYPE = Union[NoneType, pd.DataFrame, pd.Series, NoneType]


def _groups_list_by_columns(df, groups):
    tlist = [tuple()]
    if len(groups) == 0:
        return tlist

    for g in groups:
        values = list(df[g].unique())

        tvlist = []
        for t in tlist:
            for v in values:
                tvlist.append(t + (v,))

        tlist = tvlist
    # end
    return tlist
# end


def _groups_list_by_index(df):
    assert isinstance(df.index, pd.MultiIndex)
    index = list(df.index)
    groups = set()
    for idx in index:
        groups.add(idx[0:-1])

    return list(groups)
# end


def groups_list(df: pd.DataFrame, groups: Union[None, str, list[str]] = None, sort=True) -> list[tuple]:
    """
    Compose the list of tuples that represent the groups

    :param df: DataFrame to split
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

    if sorted:
        glist = list(sorted(glist))

    return glist
# end


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


def groups_split(df: pd.DataFrame, *, groups: Union[None, str, list[str]] = None, drop=True, keep=-1) \
        -> dict[tuple[str], pd.DataFrame]:
    """
    Split the dataframe based on the content of 'group' columns list or the MultiIndex.

    If 'groups' is None or the empty list AND df has a norma index, it is returned a dictionary
    with key the 'empty tuple' '()'

    If the dataset has as index a MultIndex, it is split based on the first 'n-levels-1'.
    The last level must be a PeriodIndex

    :param df: DataFrame to split
    :param groups: list of columns to use during the split. The columns must be categorical or string.
        None if it is used the MultiIndex
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

def as_tuple(l):
    t = type(l)
    if t == tuple:
        return l
    if t == list:
        return tuple(l)
    else:
        return (l,)


def _group_select_by_columns(df, groups, values, drop):
    groups = as_list(groups)
    values = as_list(values)

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


def _groups_select_by_index(df, values, drop):
    # if 'drop=True', the index levels change!
    # default behavior of 'df.loc[...]'
    values = as_tuple(values)

    if None not in values and drop:
        return df.loc[values]

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


def groups_select(df: pd.DataFrame, *,
                  values: Union[None, str, list[str], tuple[str]],
                  groups: Union[None, str, list[str], tuple[str]] = None,
                  drop=True):
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

    if groups is None:
        return _groups_select_by_index(df, values, drop)
    else:
        return _group_select_by_columns(df, groups, values, drop)
# end


# dataframe_split_on_groups = groups_split
# dataframe_merge_on_groups = groups_merge


# ---------------------------------------------------------------------------
# split_column
# columns_split
# columns_merge
# ---------------------------------------------------------------------------

def split_column(df: pd.DataFrame,
                 column: str,
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


def columns_split(df: pd.DataFrame,
                  columns: Union[None, str, list[str]] = None,
                  ignore: Union[None, str, list[str]] = None) \
        -> list[pd.Series]:
    """
    Split the dataframe in a list of series based on the list of selected columns
    """
    assert isinstance(df, pd.DataFrame)
    columns = as_list(columns, 'columns')
    ignore = as_list(ignore, 'ignore')

    if len(columns) == 0:
        columns = list(df.columns.difference(ignore))

    series = []
    for col in columns:
        series.append(df[col])

    return series
# end


def columns_merge(df: pd.DataFrame,
                  columns: list[str],
                  column: str,
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


# dataframe_split_column = split_column
# dataframe_split_on_columns = columns_split
# dataframe_merge_columns = columns_merge


# ---------------------------------------------------------------------------
# multiindex_get_level_values
# dataframe_index
# dataframe_split_on_index
# dataframe_merge_on_index
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


def set_index(df: pd.DataFrame,
              index: Union[None, str, list[str]],
              inplace=False,
              drop=False) -> pd.DataFrame:
    """
    Create a multiindex based on the columns list

    :param df: dataframe to process
    :param index: column or list of columns to use in the index
    :param inplace: if to apply the transformation inplace
    :param drop: if to drop the columns
    :return: the new dataframe
    """
    if inplace:
        df.set_index(index, inplace=inplace, drop=drop)
    else:
        df = df.set_index(index, inplace=inplace, drop=drop)
    return df
# end


# dataframe_index = set_index


def index_split(df: pd.DataFrame, levels: int = -1, drop=True) -> dict[tuple, pd.DataFrame]:
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


def index_merge(dfdict: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
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
        mi = pd.MultiIndex.from_tuples(tuples)
        dflv.set_index(mi, inplace=True)
        dflist.append(dflv)
    # end
    df = pd.concat(dflist, axis=0)
    return df
# end


# dataframe_split_on_index = index_split
# dataframe_merge_on_index = index_merge


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
    :param shared: the columns mus be present in 'X' and 'y'
    :return: list of split dataframes
    """
    target = as_list(target, 'target')
    shared = as_list(shared, 'shared')

    xy_list = []
    for data in data_list:
        assert isinstance(data, pd.DataFrame)

        columns = data.columns.difference(target).union(shared)
        X = data[columns]

        columns = data.columns.difference(columns).union(target).union(shared)
        y = data[columns]

        xy_list += [X, y]
    # end
    return xy_list
# end


def nan_split(*data_list,
              ignore: Union[None, str, list[str]] = None,
              columns: Union[None, str, list[str]] = None,
              empty=True) -> list[PANDAS_TYPE]:
    """
    Remove the columns of the dataframe containing nans

    :param data_list: list of dataframe to analyze
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

def _train_test_split_single(data, train_size, test_size) -> tuple[pd.DataFrame, pd.DataFrame]:
    def _train_size(data) -> int:
        n = len(data)
        tsize = train_size
        if 0 < test_size < 1:
            tsize = 1 - test_size
        elif test_size > 1:
            tsize = n - test_size
        if 0 < tsize < 1:
            return int(tsize * n)
        elif tsize > 1:
            return tsize
        else:
            return 1

    t = _train_size(data)
    trn = data[:t]
    tst = data[t:]
    return trn, tst
# end


def _train_test_split_multiindex(data, train_size, test_size) -> tuple[pd.DataFrame, pd.DataFrame]:
    d_data = index_split(data, -1)
    d_trn = {}
    d_tst = {}
    for lv in d_data:
        dlv = d_data[lv]
        trnlv, tstlv = _train_test_split_single(dlv, train_size, test_size)
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


def train_test_split(*data_list, train_size=0, test_size=0,
                     groups: Union[None, str, list[str]] = None,
                     sortby: Union[None, str, list[str]] = None) -> list[PANDAS_TYPE]:
    """
    Split the df in train/test
    If df has a MultiIndex, it is split each sub-dataframe based on the first [n-1]
    levels
    
    It is possible to specify 'train_size' or 'test_size', not both!
    The value of 'train_size'/'test_size' can be a number in range [0,1] or greater than 1
    
    :param data_list: 
    :param float train_size: train size
    :param float test_size: test size 
    :return: 
    """
    assert train_size > 0 or test_size > 0, "train_size or test_size must be > 0"

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
        assert data is not None

        if len(groups) > 0:
            dfdict = groups_split(data, groups=groups)
            d_trn = {}
            d_tst = {}
            for key in dfdict:
                df = dfdict[key]
                df = _sort_by(df, sortby=sortby)
                trn, tst = _train_test_split_single(df, train_size=train_size, test_size=test_size)
                d_trn[key] = trn
                d_tst[key] = tst
            # end
            trn = groups_merge(d_trn, groups=groups)
            tst = groups_merge(d_tst, groups=groups)
        elif type(data.index) == pd.MultiIndex:
            trn, tst = _train_test_split_multiindex(data, train_size=train_size, test_size=test_size)
        else:
            trn, tst = _train_test_split_single(data, train_size=train_size, test_size=test_size)
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


def series_range(df: pd.DataFrame, col: Union[str, int], dx: float = 0) -> tuple:
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
# dataframe_clip_outliers
# ---------------------------------------------------------------------------

def dataframe_correlation(df: Union[DATAFRAME_OR_DICT],
                          columns: Union[str, list[str]],
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
                  columns: Union[str, list[str]],
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


# dataframe_clip_outliers = clip_outliers
# dataframe_filter_outliers = filter_outliers


# ---------------------------------------------------------------------------
# Index labels
# ---------------------------------------------------------------------------

def index_labels(data: Union[pd.DataFrame, pd.Series], n_labels: int = -1) -> list[str]:
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
# dataframe_ignore
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


def ignore_columns(df: pd.DataFrame, ignore: Union[str, list[str]]) -> pd.DataFrame:
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


# dataframe_ignore = ignore_columns


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
