from typing import List
import pandas as pd
import arff
from .base import datetime_encode, onehot_encode, \
    dataframe_index, dataframe_ignore, datetime_reindex, _as_list, \
    unnamed_columns
from .time import periodic_encode


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
              categorical=None,
              boolean=None,
              numeric=None,
              index=None,
              ignore=None,
              ignore_unnamed=False,
              onehot=None,
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
    assert isinstance(datetime, (type(None), str, list, tuple)), \
        "'datetime' must be (None, str, (str, str), (str, str, str))"
    assert isinstance(periodic, (type(None), str, list, tuple)), \
        "'periodic' must be (None, str, (str, str), (str, dict))"
    assert isinstance(count, bool), "'count' bool"

    categorical = _as_list(categorical, 'categorical')
    boolean = _as_list(boolean, 'boolean')
    numeric = _as_list(numeric, 'numeric')
    index = _as_list(index, 'index')
    ignore = _as_list(ignore, 'ignore')
    onehot = _as_list(onehot, 'onehot')

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
    # elif file.endswith(".arff"):
    #     df = read_arff(file, dtype=dt, **kwargs)
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

    if ignore_unnamed:
        ignore += unnamed_columns(df)

    if len(ignore) > 0:
        df = dataframe_ignore(df, ignore)

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
