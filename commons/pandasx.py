#
# Pandas reader
# Some simple data set loaders: from csv and .arff data files
#
from typing import List, AnyStr, Union

import arff
import pandas as pd
import numpy as np
from math import isnan


# ---------------------------------------------------------------------------
# read_arff
# load_data == read_data
# ---------------------------------------------------------------------------

def read_arff(file, **args):
    def _parse_bool(s, default=False):
        if s is None:
            return default
        if type(s) == str:
            assert type(s) == str
            s = s.lower()
        if s in [1, "true", "on", "open", "1"]:
            return True
        if s in [0, "false", "off", "close", "0", ""]:
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
        else _parse_bool(args.get("category"))
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

def load_data(file: str, categorical=[], boolean=[], dtype=None, **args) -> pd.DataFrame:
    """
    Read the dataset from a file and convert it in a Pandas DataFrame.
    It uses the correct 'read' function based on the file extensions.

    Added the support for '.arff' file format

    :param file: file to read
    :param list categorical: list of categorical fields
    :param list boolean: list of boolean fields
    :param dict args: extra parameters passed to pd.read_xxx()
    :return pd.DataFrame: a Pandas DataFrame
    """
    if file is None:
        raise TypeError("expected str, bytes or os.Path like object, not NoneType")

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

    if dtype is not None:
        categorical, boolean = _parse_dtype(list(df.columns), dtype)

    for col in categorical:
        df[col] = df[col].astype('category')

    for col in boolean:
        df[col] = df[col].astype(bool)

    if 'count' not in df:
        df['count'] = 1.

    print("... done!")
    return df


read_data = load_data


# ---------------------------------------------------------------------------
# One hot encoding
# ---------------------------------------------------------------------------

def onehot_encode(data: pd.DataFrame, columns: List[AnyStr]=[]) -> pd.DataFrame:
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

# ---------------------------------------------------------------------------
# Cumulant, Lift
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


def cumulant(df: pd.DataFrame, select: list) -> pd.DataFrame:
    """
            cumulant(f1,..fk) = prob(f1,..fk) - (prob(f1)*...*prob(fk))

    :param df:
    :param select:
    :return:
    """
    index, cvalues, lvalues = _lift_cumulant(df, select)
    return pd.DataFrame(data={"cumulant": pd.Series(cvalues, index=index, name="cumulant")})


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


def prob(df: pd.DataFrame, select: list) -> pd.Series:
    if 'count' not in df:
        df['count'] = 1.
    total = df['count'].count()
    gcount = df[select + ['count']].groupby(select).count()/total

    return gcount
