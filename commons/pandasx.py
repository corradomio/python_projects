#
# Pandas reader
# Some simple data set loaders: from csv and .arff data files
#
import arff
import pandas as pd
import numpy as np
import warnings

from typing import List, AnyStr, Union, Optional
from math import isnan, sqrt


# ---------------------------------------------------------------------------
# read_arff
# read_data
# ---------------------------------------------------------------------------

def read_arff(file, **args):
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
# read_data
# ---------------------------------------------------------------------------

def read_data(file: str,
              *,
              dtype=None,
              categorical=[],
              boolean=[],
              index=[],
              ignore=[],
              onehot=[],
              datetime=None,
              count=False,
              dropna=True,
              **args) -> pd.DataFrame:
    """
    Read the dataset from a file and convert it in a Pandas DataFrame.
    It uses the correct 'read' function based on the file extensions.

    Added the support for '.arff' file format

    :param file: file to read
    :param categorical: list of categorical columns
    :param boolean: list of boolean columns
    :param dtype: list of column types
    :param datetime: colum used as datetime. Formats:
                None
                col: column already in datetime format
                (col, format): 'format' used to convert the string in datetime
                (col, format, freq)
    :param index: column or list of columns to use as index
    :param ignore: column or list of columns to ignore
    :param onehot: columns to convert using onehot encoding
    :param count: if to add the column 'count' with value 1
    :param dropna: if to drop rows containing NA values
    :param dict args: extra parameters passed to pd.read_xxx()
    :return pd.DataFrame: a Pandas DataFrame
    """
    # if file is None:
    #     raise TypeError("expected str, bytes or os.Path like object, not NoneType")
    assert isinstance(file, str), "'file' must be a str"
    assert isinstance(categorical, list), "'categorical' must be a list[str]"
    assert isinstance(boolean, list), "'boolean' must be a list[str]"
    assert isinstance(index, list), "'index' must be a list[str]"
    assert isinstance(ignore, list), "'ignore' must be a list[str]"
    assert isinstance(onehot, list), "'onehot' must be a list[str]"
    assert isinstance(datetime, (type(None), str, list, tuple)), "'datetime' must be (None, str, list, tuple)"
    assert isinstance(count, bool), "'count' bool"

    dt = None
    if dtype is not None:
        h = _read_header(file)
        dt = _pandas_dtype(h, dtype)
    # end

    print("Loading {} ...".format(file))

    if file.endswith(".csv"):
        df = pd.read_csv(file, dtype=dt, **args)
    elif file.endswith(".json"):
        df = pd.read_json(file, dtype=dt, **args)
    elif file.endswith(".html"):
        df = pd.read_html(file, dtype=dt, **args)
    elif file.endswith(".xls"):
        df = pd.read_excel(file, dtype=dt, **args)
    elif file.endswith(".xlsx"):
        df = pd.read_excel(file, dtype=dt, **args)
    elif file.endswith(".hdf"):
        df = pd.read_hdf(file, dtype=dt, **args)
    elif file.endswith(".arff"):
        df = read_arff(file, dtype=dt, **args)
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

    if count and 'count' not in df.columns:
        df['count'] = 1.

    if datetime is not None:
        df = datetime_encode(df, datetime)
        
    if len(onehot) > 0:
        df = onehot_encode(df, onehot)

    if len(index) > 0:
        df = dataframe_index(df, index)

    if len(ignore) > 0:
        df = dataframe_ignore(df, ignore)

    print(f"... done ({df.shape})")
    return df
# end


# ---------------------------------------------------------------------------
# One hot encoding
# ---------------------------------------------------------------------------

def onehot_encode(data: pd.DataFrame, columns: List[Union[str, int]]=[]) -> pd.DataFrame:
    """
    Add some columns based on One-Hot encode
    :param pd.DataFrame data:
    :param list[str] columns: list of columns to convert
    :return pd.DataFrame: new dataframe
    """
    for col in columns:
        dummies = pd.get_dummies(data[col], prefix=col)
        data = data.join(dummies)
    return data
# end


def datetime_encode(df: pd.DataFrame, 
                    datetime: tuple[str], 
                    format: Optional[str] = None, 
                    freq: Optional[str] = None):
    assert isinstance(datetime, (str, list, tuple))
    assert isinstance(format, (type(None), str))
    assert isinstance(freq, (type(None), str))
    assert 1 < len(datetime) < 4
    if len(datetime) == 2:
        datetime, format = datetime
    else:
        datetime, format, freq = datetime
    
    if format is not None:
        df[datetime] = pd.to_datetime(df[datetime], format=format)
    if freq is not None:
        df[datetime] = df[datetime].dt.to_period(freq)
    return df
# end


# ---------------------------------------------------------------------------
# Statistical infer_freq
# ---------------------------------------------------------------------------
# B         business day frequency
# C         custom business day frequency
# D         calendar day frequency
# W         weekly frequency
# M         month end frequency
# SM        semi-month end frequency (15th and end of month)
# BM        business month end frequency
# CBM       custom business month end frequency
# MS        month start frequency
# SMS       semi-month start frequency (1st and 15th)
# BMS       business month start frequency
# CBMS      custom business month start frequency
# Q         quarter end frequency
# BQ        business quarter end frequency
# QS        quarter start frequency
# BQS       business quarter start frequency
# A, Y      year end frequency
# BA, BY    business year end frequency
# AS, YS    year start frequency
# BAS, BYS  business year start frequency
# BH        business hour frequency
# H         hourly frequency
# T, min    minutely frequency
# S         secondly frequency
# L, ms     milliseconds
# U, us     microseconds
# N         nanoseconds

def infer_freq(index, steps=5, ntries=3) -> str:
    """
    Infer 'freq' checking randomly different positions
    of the index

    :param index:
    :param steps:
    :param ntries:
    :return:
    """
    n = len(index)-steps
    freq = None
    itry = 0
    while itry < ntries:
        i = random.randrange(n)
        tfreq = pd.infer_freq(index[i:i+steps])
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


# ---------------------------------------------------------------------------
# Series argmax
# ---------------------------------------------------------------------------

def series_argmax(df: pd.DataFrame, col: Union[str, int], argmax_col: Union[str, int]):
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


def series_argmin(df: pd.DataFrame, col: Union[str, int], argmin_col: Union[str, int]):
    """
    Let df a dataframe, search the row in 'argmin_col' with the lowest value
    then extract from 'col' the related value

    :param df: database 
    :param col: columns where to extract the value
    :param argmin_col: column where to search the minimum value
    :return: 
    """
    s = df[argmax_col]
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


def series_range(df: pd.DataFrame, col: Union[str, int], dx: float=0) -> tuple:
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
# DataFrame utilities
# ---------------------------------------------------------------------------

DATAFRAME_OR_DICT = Union[pd.DataFrame, dict[tuple, pd.DataFrame]]
TRAIN_TEST_TYPE = tuple[DATAFRAME_OR_DICT, Union[None, DATAFRAME_OR_DICT]]
PANDAS_TYPE = Union[pd.DataFrame, pd.Series]


def dataframe_ignore(df: pd.DataFrame, ignore: Union[str, list[str]]) -> pd.DataFrame:
    if isinstance(ignore, str):
        ignore = [ignore]
    assert isinstance(ignore, (list, tuple))
    return df[df.columns.difference(ignore)]
# end


def dataframe_split_on_groups(
        df: pd.DataFrame,
        groups: Union[None, str, list[str]],
        drop=False) -> dict[tuple[str], pd.DataFrame]:
    """
    Split the dataframe based on the content area columns list

    :param df: DataFrame to split
    :param groups: list of columns to use during the split. The columns must be categorical
    :param ignore: if to remove the 'groups' columns'

    :return: a list [((g1,...), gdf), ...]
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
            dfdict[g] = gdf[gdf.columns.difference(groups)]
    # end

    return dfdict
# end


def dataframe_merge_on_groups(dfdict: dict[tuple[str], pd.DataFrame],
                              groups: Union[str, list[str]],
                              sortby: Union[None, str, list[str]] = None) \
        -> pd.DataFrame:
    """

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


def dataframe_split_on_columns(df: pd.DataFrame,
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


# ---------------------------------------------------------------------------
# Dataframe index utilities
# ---------------------------------------------------------------------------

def multiindex_get_level_values(mi: Union[pd.DataFrame, pd.Series, pd.MultiIndex], levels=1) -> list[tuple]:
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
    if inplace:
        df.set_index(index, inplace=inplace, drop=drop)
    else:
        df = df.set_index(index, inplace=inplace, drop=drop)
    return df
# end


def dataframe_split_on_index(df: PANDAS_TYPE, levels: int = 1) -> dict[tuple, pd.DataFrame]:
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


def dataframe_merge_on_index(dfdict: dict[tuple, pd.DataFrame]) -> pd.DataFrame:
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


# ---------------------------------------------------------------------------
# Split utilities
# ---------------------------------------------------------------------------

def xy_split(*data_list, target: Union[str, list[str]]) -> list[PANDAS_TYPE]:
    """
    Split the df in 'data_list' in X, y
    
    :param data_list: df list
    :param target: target column name
    :return: list of splitte df
    """
    assert isinstance(target, (str, list))
    Xy_list = []
    for data in data_list:
        assert isinstance(data, pd.DataFrame)
        if isinstance(target, str):
            X = data[data.columns.difference([target])]
            y = data[[target]]
            Xy_list += [X, y]
        else:
            X = data[data.columns.difference(target)]
            Y = data[[target]]
            Xy_list += [X, y]
    # end
    return Xy_list
# end


def train_test_split(*data_list, train_size=0, test_size=0) -> list[PANDAS_TYPE]:
    """
    Split the df in train/test
    If df has a MultiIndex, it is splitted each sub-dataframe based on the first [n-1]
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
    gcount = df[select + ['count']].groupby(select).count()/total

    # single count
    scount = dict()
    for c in select:
        scount[c] = df[[c] + ['count']].groupby([c]).count()/total

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

        lvalue = gvalue/sproduct if sproduct != 0. else 0.
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
    gcount = df[select + ['count']].groupby(select).count()/total

    return gcount
# end


# ---------------------------------------------------------------------------
# partition_lengths
# partitions_split
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


def partitions_split(*data_list : list[pd.DataFrame], partitions: Union[int, list[int]]=1, index=None, random=False) \
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
# ---------------------------------------------------------------------------

def to_dataframe(data: np.ndarray, *, target: Union[str, list[str]], index=None) -> pd.DataFrame:
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
    t = sqrt(c)/c
    # create the result data structure
    cq = pd.DataFrame({}, index=pred_proba.index)
    # classification quality
    cq[target] = (np.linalg.norm(pred_proba.values, axis=1) - t)/(1 - t)
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
