from typing import List
import pandas as pd
from stdlib import NoneType
from .base import datetime_encode, onehot_encode, binary_encode, \
    set_index, ignore_columns, datetime_reindex, as_list, \
    find_unnamed_columns, find_binary
from .time import periodic_encode


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
# dtype,
# categorical, boolean, numeric, onehot, binary,
# datetime, periodic,
# index, reindex,
# ignore, ignore_unnamed,
# count,
# na_values, dropna,
#

def read_data(file: str,
              *,
              dtype=None,
              categorical=None,
              boolean=None,
              numeric=None,
              onehot=None,
              binary=None,

              index=None,
              ignore=None,
              ignore_unnamed=False,

              datetime=None,
              periodic=None,

              count=False,
              reindex=False,

              na_values=None,
              dropna=False,
              **kwargs) -> pd.DataFrame:
    """
    Read the dataset from a file and convert it in a Pandas DataFrame.
    It uses the correct 'read' function based on the file extensions.

    Added support for '.arff' file format

    Note: parameters for '.csv' files:

        sep             str, default ','
        delimiter       alias for sep
        header          int, list[int], None, default 'infer'
        names           array-like, optional
        index_col       int, str, sequence of int / str, or False, optional, default None
        usecol          list-like or callable, optional
        dtype           Type name or dict of column -> type, optional
        true_values     list, optional
        false_values    list, optional
        skipinitialspace    bool, default False
        skiprows        list-like, int or callable, optional
        skipfooter      int, default 0
        nrows           int, optional
        na_vales        scalar, str, list-like, or dict, optional
        keep_default_na bool, default True
        na_filter       bool, default True
        skip_blank_lines    bool, default True
        parse_dates     bool or list of int or names or list of lists or dict, default False
        keep_date_col   bool, default False
        date_format     str or dict of column -> format, default None
        dayfirst        bool, default False
        ...

    for other parameters, see 'pandas.read_csv()' documentation

    :param file: file to read
    :param dtype: list of column types. A column type can be:

                - None: column skipped
                - str: string
                - int: integer
                - float: float
                - bool: boolean value in form 0/1, off/on, close/open/ f/t, false/true
                    (case insensitive)
                - enum: string used as enumeration
                - ienum: integer value used as enumeration
    :param categorical: columns to convert in 'pandas.categorical' format
    :param boolean: columns to convert in 'boolean' type
    :param numeric: columns to convert in 'float' type. To force a float value on integer columns
    :param datetime: column used as datetime. Formats:
                None
                col: column already in datetime format
                (col, format): 'format' used to convert the string into pandas datetime
                (col, format, freq): 'freq' used to convert a datetime in a pandas period
    :param index: column or list of columns to use as index. With multiple columns, it is created
                a MultiIndex following the columns order
    :param ignore: column or list of columns to ignore
    :param ignore_unnamed: if to ignore 'Unnamed: *' columns
    :param onehot: columns to convert using onehot encoding
    :param count: if to add the column 'count' with value 1
    :param dropna: if to drop rows containing NA values
    :param periodic: if to add 'periodic' information (EXPERIMENTAL)
    :param reindex: if to 'reindex' the dataframe in such way to force ALL timestamp (EXPERIMENTAL)
    :param dict kwargs: extra parameters passed to pd.read_xxx()
    :return pd.DataFrame: a Pandas DataFrame
    """
    assert isinstance(file, str), "'file' must be a str"
    assert isinstance(datetime, (NoneType, str, list, tuple)), \
        "'datetime' must be (None, str, (str, str), (str, str, str))"
    assert isinstance(periodic, (NoneType, str, list, tuple)), \
        "'periodic' must be (None, str, (str, str), (str, dict))"
    assert isinstance(count, bool), "'count' bool"

    categorical = as_list(categorical, 'categorical')
    boolean = as_list(boolean, 'boolean')
    numeric = as_list(numeric, 'numeric')
    onehot = as_list(onehot, 'onehot')
    binary = as_list(binary, 'binary')
    ignore = as_list(ignore, 'ignore')
    index = as_list(index, 'index')

    # move 'na_values' in kwargs
    if na_values is not None:
        kwargs['na_values'] = na_values

    # if dtype is defined, compose the dictionary {col:dtype}
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
        df = pd.read_html(file, **kwargs)
    elif file.endswith(".xls"):
        df = pd.read_excel(file, dtype=dt, **kwargs)
    elif file.endswith(".xlsx"):
        df = pd.read_excel(file, dtype=dt, **kwargs)
    elif file.endswith(".hdf"):
        df = pd.read_hdf(file, dtype=dt, **kwargs)
    # elif file.endswith(".arff"):
    #     df = read_arff(file, dtype=dt, **kwargs)
    elif "://" in file:
        df = read_database(file, dtype=dt, **kwargs)
    else:
        raise TypeError("File extension unsupported: " + file)

    if dtype is not None:
        categorical, boolean = _parse_dtype(list(df.columns), dtype)

    if binary == ['auto']:
        binary = find_binary(df, onehot)
        onehot = set(onehot).difference(binary)

    # pandas categorical
    for col in categorical:
        df[col] = df[col].astype('category')

    # pandas boolean
    for col in boolean:
        df[col] = df[col].astype(bool)

    # pandas float
    for col in numeric:
        df[col] = df[col].astype(float)

    # add the 'count' column
    if count and 'count' not in df.columns:
        df['count'] = 1.

    # convert the datetime column
    if datetime is not None:
        df = datetime_encode(df, datetime)

    # encode the periodic column
    if periodic is not None:
        df = periodic_encode(df, *periodic)

    # onehot encoding
    if len(onehot) > 0:
        df = onehot_encode(df, onehot)

    # binary/{0,1} encoding
    if len(binary) > 0:
        df = binary_encode(df, binary)

    # compose the index
    if len(index) > 0:
        df = set_index(df, index)

    # add 'Unnamed: ...' columns to the list of columns to remove
    if ignore_unnamed:
        ignore += find_unnamed_columns(df)

    # remove the 'ignore' columns
    if len(ignore) > 0:
        df = ignore_columns(df, ignore)

    # remove the rows containing NaN
    # note: it is better to check for NaN ONLY AFTER ignored columns
    if dropna:
        df = df.dropna(how='any', axis=0)

    # force the reindex
    if reindex:
        df = datetime_reindex(df)

    print(f"... done ({df.shape})")
    return df
# end


def write_data(df: pd.DataFrame, file: str, **kwargs):
    if file.endswith("csv"):
        if "index" not in kwargs:
            kwargs['index'] = False
        df.to_csv(file, **kwargs)
    elif file.endswith(".json"):
        df.to_json(file, **kwargs)
    elif file.endswith(".xml"):
        df.to_xml(file, **kwargs)
    else:
        raise ValueError(f"Unsupported file format {file}")
# end
