#
# Pandas reader
# Some simple data set loaders: from csv and .arff data files
#
import arff
import pandas as pd
import numpy as np
import warnings
import random
import math

from typing import List, AnyStr, Union, Optional
from math import isnan, sqrt
from random import random


# ---------------------------------------------------------------------------
# read_arff
# ---------------------------------------------------------------------------

def read_arff(file, **args):
    """
    Read an ARFF file, a CSV like text file with format specified in

        https://www.cs.waikato.ac.nz/~ml/weka/arff.html

    based on the library

        https://pythonhosted.org/liac-arff/
        https://pypi.org/project/liac-arff/2.2.1/


    :param file: file to load
    :param args: arguments passed to 'liac-arff' library
    :return:
    """
    def _tobool(s, default=False):
        if s is None:
            return default
        if type(s) == str:
            s = s.lower()
        assert isinstance(s, (bool, str, int))
        if s in [1, True, "true", "on", "open", "1"]:
            return True
        if s in [0, False, "false", "off", "close", "0", ""]:
            return False
        return default

    fdict = arff.load_file(file, **args)
    alist = fdict['attributes']
    """:type: list[tuple[str, list|str]]"""
    data = fdict['data']
    """:type: list[list]"""
    names = list(map(lambda a: a[0], alist))
    """:type: list[str]"""
    df = pd.DataFrame(data, columns=names)
    """:type: pd.DataFrame"""

    category = True if "category" not in args \
        else _tobool(args.get("category"))
    """:type: bool"""

    if category:
        for attr in alist:
            aname = attr[0]
            atype = type(attr[1])
            if atype == list:
                df[aname] = df[aname].astype('category')
    return df
# end

def _parse_dtype(columns, dtype):
    categorical = []
    boolean = []

    i = -1
    for t in dtype:
        i += 1
        if t in [bool, "bool", "boolean"]:
            boolean.append(columns[i])
        if t in ["senum", "enum", "ienum", enumerate]:
            categorical.append(columns[i])
    # end
    return categorical, boolean
# end

def _read_header(file: str, comment="#", sep=",") -> List[str]:
    def trim(s: str) -> str:
        return s.strip(" '\"")
    with open(file) as fin:
        line = comment
        while line.startswith(comment):
            line = next(fin)
        return list(map(trim, line.split(sep)))
    # end
# end

def _pandas_dtype(columns, dtype) -> dict:
    assert len(columns) == len(dtype)
    dt = dict()
    for i in range(len(columns)):
        ct = dtype[i]
        if ct == enumerate:
            dt[columns[i]] = str
        else:
            dt[columns[i]] = ct
    return dt
# end

# ---------------------------------------------------------------------------
# read_database
# ---------------------------------------------------------------------------
# protocol://host:port/database?table=...
# protocol://host:port/database?sql=...
# protocol://host:port/database/table

def _to_url_select(url: str):
    p = url.find('?')
    if p == -1:
        # .../table
        p = url.rfind('/')
        table = url[p + 1:]
        url = url[0:p]
        return url, f'select * from {table}'

    # ...?table=...
    # ...?sql=select ...
    what = url[p + 1:].strip()
    url = url[0:p]
    if what.startswith('table='):
        what = what[6:]
    elif what.startswith('sql='):
        what = what[4:]
    else:
        raise ValueError(f'Invalid url {url}')
    p = what.find(' ')
    # what: table
    # what: 'select ....'
    if p == -1:
        return url, f'select * from {what}'
    else:
        return url, what
# end


def read_database(url: str, dtype, **kwargs):
    url, sql = _to_url_select(url)
    from sqlalchemy import create_engine
    engine = create_engine(url)
    with engine.connect() as con:
        df = pd.read_sql(sql=sql, con=con, params=kwargs)
    return df
# end


# ---------------------------------------------------------------------------
# read_data
# ---------------------------------------------------------------------------

def read_data(file: str,
              *,
              dtype=None,
              categorical=[],
              boolean=[],
              numeric=[],
              index=[],
              ignore=[],
              onehot=[],
              datetime=None,
              periodic=None,
              count=False,
              dropna=False,
              reindex=False,
              na_values=None,
              **kwargs) -> pd.DataFrame:
    """
    Read the dataset from a file and convert it in a Pandas DataFrame.
    It uses the correct 'read' function based on the file extensions.

    Added support for '.arff' file format

    :param file: file to read
    :param categorical: list of categorical columns
    :param boolean: list of boolean columns
    :param dtype: list of column types. A column type can be:

                - None: column skipped
                - str: string
                - int: integer
                - float: float
                - bool: boolean value in form 0/1, off/on, close/open/ f/t, false/true
                    (case insensitive)
                - enum: string used as enumeration
                - ienum: integer value used as enumeration

    :param datetime: column used as datetime. Formats:
                None
                col: column already in datetime format
                (col, format): 'format' used to convert the string in datetime
                (col, format, freq): 'freq' used to convert a datetim in a perio
    :param index: column or list of columns to use as index
    :param ignore: column or list of columns to ignore
    :param onehot: columns to convert using onehot encoding
    :param categorical: columns to convert in 'pandas.categorical' format
    :param boolean: columns to convert in 'boolean' type
    :param count: if to add the column 'count' with value 1
    :param dropna: if to drop rows containing NA values
    :param dict kwargs: extra parameters passed to pd.read_xxx()
    :return pd.DataFrame: a Pandas DataFrame
    """
    # if file is None:
    #     raise TypeError("expected str, bytes or os.Path like object, not NoneType")
    assert isinstance(file, str), "'file' must be a str"
    assert isinstance(categorical, list), "'categorical' must be a list[str]"
    assert isinstance(boolean, list), "'boolean' must be a list[str]"
    assert isinstance(numeric, list), "'numeric' must be a list[str]"
    assert isinstance(index, list), "'index' must be a list[str]"
    assert isinstance(ignore, list), "'ignore' must be a list[str]"
    assert isinstance(onehot, list), "'onehot' must be a list[str]"
    assert isinstance(datetime, (type(None), str, list, tuple)), "'datetime' must be (None, str, (str, str), (str, str, str))"
    assert isinstance(periodic, (type(None), str, list, tuple)), "'periodic' must be (None, str, (str, str))"
    assert isinstance(count, bool), "'count' bool"

    # move 'na_values' in kwargs
    if na_values is not None:
        kwargs['na_values'] = na_values

    if dtype is not None:
        h = _read_header(file)
        dt = _pandas_dtype(h, dtype)
    else:
        dt = None

    print("Loading {} ...".format(file))

    if file.endswith(".csv"):
        df = pd.read_csv(file, dtype=dt, **kwargs)
    elif file.endswith(".json"):
        df = pd.read_json(file, dtype=dt, **kwargs)
    elif file.endswith(".html"):
        df = pd.read_html(file, dtype=dt, **kwargs)
    elif file.endswith(".xls"):
        df = pd.read_excel(file, dtype=dt, **kwargs)
    elif file.endswith(".xlsx"):
        df = pd.read_excel(file, dtype=dt, **kwargs)
    elif file.endswith(".hdf"):
        df = pd.read_hdf(file, dtype=dt, **kwargs)
    elif file.endswith(".arff"):
        df = read_arff(file, dtype=dt, **kwargs)
    elif "://" in file:
        df = read_database(file, dtype=dt, **kwargs)
    else:
        raise TypeError("File extension unsupported: " + file)

    if dropna:
        df = df.dropna(how='any', axis=0)

    if dtype is not None:
        categorical, boolean = _parse_dtype(list(df.columns), dtype)

    for col in categorical:
        df[col] = df[col].astype('category')

    for col in boolean:
        df[col] = df[col].astype(bool)

    for col in numeric:
        df[col] = df[col].astype(float)

    if count and 'count' not in df.columns:
        df['count'] = 1.

    if datetime is not None:
        df = datetime_encode(df, datetime)

    if periodic is not None:
        df = periodic_encode(df, *periodic)
        
    if len(onehot) > 0:
        df = onehot_encode(df, onehot)

    if len(index) > 0:
        df = dataframe_index(df, index)

    if len(ignore) > 0:
        df = dataframe_ignore(df, ignore)

    if reindex:
        df = dataframe_datetime_reindex(df)

    print(f"... done ({df.shape})")
    return df
# end


# ---------------------------------------------------------------------------
# onehot_encode
# datetime_encode
# dataframe_datetime_reindex
# ---------------------------------------------------------------------------

def onehot_encode(df: pd.DataFrame, columns: List[Union[str, int]] = []) -> pd.DataFrame:
    """
    Add some columns based on pandas' 'One-Hot encoding' (pd.get_dummies)

    :param pd.DataFrame df:
    :param list[str] columns: list of columns to convert
    :return pd.DataFrame: new dataframe
    """
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col)
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
    assert isinstance(format, (type(None), str))
    assert isinstance(freq, (type(None), str))
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
    df = df.reindex(index= dtrange, method=mehod)
    return df
# end


# ---------------------------------------------------------------------------
# periodic_encode
# ---------------------------------------------------------------------------

def periodic_encode(df, column, method, freq: Optional[str]=None) -> pd.DataFrame:
    if method == 'onehot':
        return onehot_encode(df, column)
    elif method == 'M' or freq == 'M':
        return _monthly_encoder(df, column)
    elif method == 'W' or freq == 'W':
        return _weekly_encoder(df, column)
    elif method == 'D' or freq == 'D':
        return _daily_encoder(df, column)
    else:
        raise ValueError(f"'Unsupported periodic_encode method '{method}/{freq}'")
# end


def _monthly_encoder(df, column):
    FACTOR = 2 * math.pi / 12
    dt = df[column]

    dtcos = dt.apply(lambda x: math.cos(FACTOR*(x.month-1)))
    dtsin = dt.apply(lambda x: math.sin(FACTOR*(x.month-1)))

    df[column + "_c"] = dtcos
    df[column + "_s"] = dtsin

    return df


def _weekly_encoder(df, column):
    FACTOR = 2 * math.pi / 7
    dt = df[column]

    dtcos = dt.apply(lambda x: math.cos(FACTOR * (x.weekday)))
    dtsin = dt.apply(lambda x: math.sin(FACTOR * (x.weekday)))

    df[column + "_c"] = dtcos
    df[column + "_s"] = dtsin

    return df


def _daily_encoder(df, column):
    FACTOR = 2 * math.pi / 24
    dt = df[column]

    dtcos = dt.apply(lambda x: math.cos(FACTOR * (x.hour)))
    dtsin = dt.apply(lambda x: math.sin(FACTOR * (x.hour)))

    df[column + "_c"] = dtcos
    df[column + "_s"] = dtsin

    return df


# ---------------------------------------------------------------------------
# dataframe_split_column
# ---------------------------------------------------------------------------

def dataframe_split_column(df: pd.DataFrame,
                           column: str,
                           columns: Optional[list[str]] = None,
                           sep: str = '~',
                           drop=False) -> pd.DataFrame:
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
        columns = [f'{col}_{j+1}' for j in range(p)]
        
    # populate the dataframe
    df = df.copy()
    for j in range(p):
        df[columns[j]] = splits[j]

    # drop the column if required
    if drop:
        df.drop(col, inplace=True)

    return df
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

def dataframe_ignore(df: pd.DataFrame, ignore: Union[str, list[str]]) -> pd.DataFrame:
    """
    Remove a column or list of columns
    :param df:
    :param ignore:
    :return:
    """
    if ignore is None:
        return df

    elif isinstance(ignore, str):
        ignore = [ignore]
    assert isinstance(ignore, (list, tuple))
    return df[df.columns.difference(ignore)]
# end


# ---------------------------------------------------------------------------
# dataframe_split_on_groups
# dataframe_merge_on_groups
# ---------------------------------------------------------------------------

DATAFRAME_OR_DICT = Union[pd.DataFrame, dict[tuple, pd.DataFrame]]
TRAIN_TEST_TYPE = tuple[DATAFRAME_OR_DICT, Union[None, DATAFRAME_OR_DICT]]
PANDAS_TYPE = Union[pd.DataFrame, pd.Series]


def groups_split(df: pd.DataFrame, groups: Union[None, str, list[str]], drop=False) \
    -> dict[tuple[str], pd.DataFrame]:
    """
    Split the dataframe based on the content of 'group' columns list.

    If 'groups' is None or the empty list, it is returned a dictionary with key
    the 'empty tuple' (a tuple of length zero)

    :param df: DataFrame to split
    :param groups: list of columns to use during the split. The columns must be categorical or string
    :param ignore: if to remove the 'groups' columns

    :return dict[tuple[str], DataFrame]: a dictionary
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(groups, (type(None), str, list))

    if isinstance(groups, str):
        groups = [groups]
    elif groups is None:
        groups = []

    if len(groups) == 0:
        return {tuple(): df}

    dfdict: dict[tuple, pd.DataFrame] = {}

    # Note: IF len(groups) == 1, Pandas return 'gname' in instead than '(gname,)'
    # The library generates a FutureWarning !!!!
    if len(groups) == 1:
        for g, gdf in df.groupby(by=groups[0]):
            dfdict[(g,)] = gdf
    else:
        for g, gdf in df.groupby(by=groups):
            dfdict[g] = gdf
    # end
    
    if drop:
        for g in dfdict:
            gdf = dfdict[g]
            # dfdict[g] = gdf[gdf.columns.difference(groups)]
            gdf.drop(groups, inplace=True, axis=1)
    # end

    return dfdict
# end


def groups_merge(dfdict: dict[tuple[str], pd.DataFrame],
                 groups: Union[str, list[str]],
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
    assert isinstance(groups, (str, list))

    if isinstance(groups, str):
        groups = [groups]
    if isinstance(sortby, str):
        sortby = [sortby]

    n = len(groups)
    dfonly = []
    for gvalues in dfdict:
        assert len(gvalues) == len(groups)
        gdf = dfdict[gvalues]
        gdf = gdf.copy()

        for i in range(n):
            gdf[groups[i]] = gvalues[i]

        dfonly.append(gdf)
    # end

    df = pd.concat(dfonly, axis=0)
    if sortby is not None:
        df.sort_values(*sortby, inplace=True)
    return df
# end


dataframe_split_on_groups = groups_split
dataframe_merge_on_groups = groups_merge


# ---------------------------------------------------------------------------
# dataframe_split_on_columns
# ---------------------------------------------------------------------------

def columns_split(df: pd.DataFrame,
                  columns: Union[None, str, list[str]] = None,
                  ignore: Union[None, str, list[str]] = None) \
        -> list[pd.Series]:
    """
    Split the dataframe in a list of series based on the list of selected columns
    """
    if ignore is None:
        ignore = []
    elif isinstance(ignore, str):
        ignore = [ignore]

    if columns is None:
        columns = list(df.columns.difference(ignore))
    elif isinstance(columns, str):
        columns = [columns]

    series = []
    for col in columns:
        series.append(df[col])

    return series
# end


dataframe_split_on_columns = columns_split


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
    :param mi: multi index or a DataFrame/Series with multi index
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


def dataframe_index(df: pd.DataFrame,
                    index: Union[None, str, list[str]],
                    inplace=False,
                    drop=False) -> pd.DataFrame:
    """
    Create a multiindex based on the list of columns

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


def index_split(df: PANDAS_TYPE, levels: int = -1) -> dict[tuple, pd.DataFrame]:
    """
    Split the dataframe based on the first 'levels' values of the multiindex

    :param df: dataframe to process
    :param levels: n of multiidex levels to consider. Can be < 0
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
    return dfdict
# end


def index_merge(dfdict: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
    """
    Recreate a dataframe using the keys in the dictionary as multiindex
    :param dfdict:
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


dataframe_split_on_index = index_split
dataframe_merge_on_index = index_merge

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
        dfdict = dataframe_split_on_groups(df, groups=groups)
        corr_dict = {}
        for key in dfdict:
            df = dfdict[key]
            dfcorr = dataframe_correlation(df, columns=columns)
            corr_dict[key] = dfcorr
        dfcorr = dataframe_merge_on_groups(corr_dict, groups=groups)
        dfcorr.set_index(dfcorr[groups], inplace=True)
        return dfcorr
    # end
    
    dftc = df[columns]
    corr = dftc.corr().loc[target]
    
    dfcorr = pd.DataFrame(columns=corr.index)
    dfcorr = dfcorr.append(corr, ignore_index=True)
    return dfcorr
# end

def clip_outliers(df: Union[DATAFRAME_OR_DICT],
                  columns: Union[str, list[str]],
                  outlier_std: float = 3,
                  groups: Union[None, str, list[str]] = None) -> pd.DataFrame:

    if isinstance(columns, str):
        columns = [columns]

    if groups is not None:
        dfdict = dataframe_split_on_groups(df, groups=groups)
        outl_dict = {}
        for key in dfdict:
            df = dfdict[key]
            dfoutl = dataframe_clip_outliers(df, columns=columns)
            outl_dict[key] = dfoutl
        dfoutl = dataframe_merge_on_groups(outl_dict, groups=groups)
        return dfoutl
    # end
    
    for col in columns:
        data = df[col].copy()
        mean = data.mean()
        std = data.std()
        median = data.median()
        
        min = mean-outlier_std*std
        max = mean+outlier_std*std
        
        data[data<min] = median
        data[data>max] = median
        df[col] = data
    # end
    return df


dataframe_clip_outliers = clip_outliers


# ---------------------------------------------------------------------------
# xy_split
# train_test_split
# nan_split
# ---------------------------------------------------------------------------

def xy_split(*data_list, target: Union[str, list[str]]) -> list[PANDAS_TYPE]:
    """
    Split the df in 'data_list' in X, y

    :param data_list: df list
    :param target: target column name
    :return: list of splitte df
    """
    if isinstance(target, str):
        target = [target]

    assert isinstance(target, list)

    xy_list = []
    for data in data_list:
        assert isinstance(data, pd.DataFrame)
        X = data[data.columns.difference(target)]
        y = data[target]
        xy_list += [X, y]
    # end
    return xy_list
# end


def train_test_split(*data_list, train_size=0, test_size=0) -> list[PANDAS_TYPE]:
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

    def _train_split(data):
        t = _train_size(data)
        return data[:t], data[t:]

    tt_list = []

    for data in data_list:
        if type(data.index) == pd.MultiIndex:
            d_data = dataframe_split_on_index(data, -1)
            d_trn = {}
            d_tst = {}
            for lv in d_data:
                dlv = d_data[lv]
                trnlv, tstlv = _train_split(dlv)
                d_trn[lv] = trnlv
                d_tst[lv] = tstlv
            # end
            trn = dataframe_merge_on_index(d_trn)
            tst = dataframe_merge_on_index(d_tst)
        else:
            trn, tst = _train_split(data)
        tt_list.append(trn)
        tt_list.append(tst)
    # end
    return tt_list
# end


def nan_split(*data_list, 
              excluding: Union[None, str, list[str]]=None,
              columns: Union[None, str, list[str]]=None) -> list[PANDAS_TYPE]:
    if isinstance(excluding, str):
        excluding = [excluding]
    if isinstance(columns, str):
        columns = [columns]

    vi_list = []

    for data in data_list:
        if excluding is not None:
            columns = data.columns.difference(excluding)
        if columns is None:
            columns = data.columns
            
        invalid_rows = data[columns].isnull().any(axis=1)
        invalid = data.loc[invalid_rows]
        valid = data.loc[data.index.difference(invalid.index)]

        vi_list += [valid, invalid]
    # end
    return vi_list
# end



# ---------------------------------------------------------------------------
# cumulant, lift, prob
# ---------------------------------------------------------------------------

def _lift_cumulant(df: pd.DataFrame, select: list) -> tuple:
    def float_(x):
        x = float(x)
        # return 0. if isnan(x) else x
        return x

    if 'count' not in df:
        df['count'] = 1.

    n = len(select)

    total = df['count'].count()

    # group count
    gcount = df[select + ['count']].groupby(select).count() / total

    # single count
    scount = dict()
    for c in select:
        scount[c] = df[[c] + ['count']].groupby([c]).count() / total

    index = gcount.index
    cvalues = []
    lvalues = []
    for keys in index.values:
        gvalue = float_(gcount.loc[keys])
        sproduct = 1.

        for i in range(n):
            c = select[i]
            k = keys[i]
            svalue = float_(scount[c].loc[k])
            if isnan(svalue): svalue = 1.
            sproduct *= svalue

        cvalue = gvalue - sproduct
        cvalues.append(cvalue)

        lvalue = gvalue / sproduct if sproduct != 0. else 0.
        lvalues.append(lvalue)
    # end
    return index, cvalues, lvalues
# end
 

def cumulant(df: pd.DataFrame, select: list) -> pd.DataFrame:
    """
            cumulant(f1,..fk) = prob(f1,..fk) - (prob(f1)*...*prob(fk))

    :param df:
    :param select:
    :return:
    """
    index, cvalues, lvalues = _lift_cumulant(df, select)
    return pd.DataFrame(data={"cumulant": pd.Series(cvalues, index=index, name="cumulant")})
# end


def lift(df: pd.DataFrame, select: list) -> pd.DataFrame:
    """
                             prob(f1,...fk)
        lift(f1,...fk) = -----------------------
                          prob(f1)*...*prob(fk)

    :param df:
    :param select:
    :return:
    """
    index, cvalues, lvalues = _lift_cumulant(df, select)
    return pd.DataFrame(data={"lift": pd.Series(lvalues, index=index, name="lift")})
# end


def prob(df: pd.DataFrame, select: list) -> pd.Series:
    if 'count' not in df:
        df['count'] = 1.
    total = df['count'].count()
    gcount = df[select + ['count']].groupby(select).count() / total

    return gcount
# end


# ---------------------------------------------------------------------------
# partition_lengths
# partitions_split
# ---------------------------------------------------------------------------

def partition_lengths(n: int, quota: Union[int, list[int]]) -> list[int]:
    if isinstance(quota, int):
        quota = [1] * quota
    k = len(quota)
    tot = sum(quota)
    lengths = []
    for i in range(k - 1):
        l = int(n * quota[i] / tot + 0.6)
        lengths.append(l)
    lengths.append(n - sum(lengths))
    return lengths
# end


def partitions_split(*data_list: list[pd.DataFrame], partitions: Union[int, list[int]] = 1, index=None, random=False) \
        -> list[Union[pd.DataFrame, pd.Series]]:
    parts_list = []
    for data in data_list:
        parts = _partition_split(data, partitions=partitions, index=index, random=random)
        parts_list.append(parts)
    # end
    parts_list = list(zip(*parts_list))
    return parts_list
# end


def _partition_split(data: pd.DataFrame, partitions: Union[int, list[int]], index, random) -> list[pd.DataFrame]:
    n = len(data)
    indices = list(range(n))
    plengths = partition_lengths(n, partitions)
    pn = len(plengths)
    s = 0
    parts = []
    for i in range(pn):
        pl = plengths[i]
        if index is None:
            part = data.iloc[s:s + pl]
        else:
            part_index = index[s:s + pl]
            part = data.loc[part_index]
        # end
        parts.append(part)
        s += pl
    # end
    return parts
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
    if isinstance(target, (tuple, list)):
        pass
    elif c == 1:
        columns = [target]
    else:
        columns = [f'{target}_{i}' for i in range(c)]

    df = pd.DataFrame(data, columns=columns, index=index)
    return df
# end


def to_numpy(data: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    if data is None:
        return None
    else:
        return data.to_numpy()
# end

# ---------------------------------------------------------------------------
# classification_quality
# ---------------------------------------------------------------------------

def classification_quality(pred_proba: Union[pd.DataFrame, pd.Series], target=None) -> pd.DataFrame:
    """
    Compute the classification quality (a number between 0 and 1) based on
    the euclidean distance, then assign an index (an integer in range [0, n))
    in such way that the best classification quality has index 0 and the worst
    index (n-1).

    :param pred_proba: the output of 'ml.pred_proba()'
    :return: an array (n, 2) where the first column contains the classification
        quality and the second columnt the quality index
    """
    assert isinstance(pred_proba, (pd.DataFrame, pd.Series))
    if target is None:
        target = 'pred_qual'

    n, c = pred_proba.shape
    t = sqrt(c) / c
    # create the result data structure
    cq = pd.DataFrame({}, index=pred_proba.index)
    # classification quality
    cq[target] = (np.linalg.norm(pred_proba.values, axis=1) - t) / (1 - t)
    # assign the original prediction indices
    # cq['origin'] = range(n)
    # order the classification qualities in desc order
    cq.sort_values(by=[target], ascending=False, inplace=True)
    # assign the quality index order
    cq['rank'] = range(n)
    # back to the original order
    cq = cq.loc[pred_proba.index]
    # remove the extra column
    # cq = cq[:, 0:2]
    # done
    return cq
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
