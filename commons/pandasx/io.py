import logging
from typing import Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text, URL

from .base import (
    set_index, nan_drop, nan_fill, as_list,
    find_unnamed_columns, dataframe_sort, columns_rename, columns_ignore,
    type_encode, count_encode,
    NoneType
)
from stdlib import dict_select, dict_exclude
from .binhot import binhot_encode
from .datetimes import datetime_encode, datetime_reindex
from .freq import infer_freq
from .io_arff import read_arff
from .onehot import onehot_encode
from .periodic import periodic_encode
from .json import from_json, to_json
from .sql import read_sql, write_sql

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _select_typed_columns(columns, dtype):
    categorical = []
    boolean = []
    ignore = []

    i = -1
    for t in dtype:
        i += 1
        if t in [bool, "bool", "boolean"]:
            boolean.append(columns[i])
        if t in ["senum", "enum", "ienum", enumerate]:
            categorical.append(columns[i])
        if t is None:
            ignore.append(columns[i])
    # end
    return categorical, boolean, ignore


def _read_header(file: str, comment="#", sep=",") -> list[str]:
    def trim(s: str) -> str:
        return s.strip(" '\"")

    with open(file) as fin:
        line = comment
        while line.startswith(comment):
            line = next(fin)
        return list(map(trim, line.split(sep)))


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


def _parse_datetime(datetime) -> tuple:
    if datetime is None:
        datetime_name, datetime_format, datetime_freq = None, None, None
    elif isinstance(datetime, str):
        datetime_name, datetime_format, datetime_freq = datetime, None, None
    elif len(datetime) == 1:
        datetime_name, datetime_format, datetime_freq = datetime[0], None, None
    elif len(datetime) == 2:
        datetime_name, datetime_format, datetime_freq = datetime[0], datetime[1], None
    elif len(datetime) == 3:
        datetime_name, datetime_format, datetime_freq = datetime
    else:
        raise ValueError("Invalid 'datetime' parameter format")
    return datetime_name, datetime_format, datetime_freq


# ---------------------------------------------------------------------------
# read_database
# ---------------------------------------------------------------------------
#   protocol://host:port/database
#       table=<table_name>
#       sql='SELECT ...'
#       params={...}
# protocol://host:port/database?table=...
# protocol://host:port/database?sql=...&p1=...&...
# protocol://host:port/database/table

# def _safe_val(x: str):
#     if isinstance(x, list):
#         return tuple(map(_safe_val, x))
#
#     try:
#         return float(x)
#     except:
#         pass
#     try:
#         return int(x)
#     except:
#         pass
#     if x.startswith("'") or x.startswith('"'):
#         return x[1:-1]
#     else:
#         return x
# # end


# def _parse_url(url: str) -> dict:
#     parts = dict(params={})
#     p = url.find('?')
#     if p != -1:
#         url = url[:p]
#         qargs = url[p+1:]
#     else:
#         qargs = ""
#
#     p = url.find("://")
#     parts["drivername"] = url[:p]
#     url = url[p+3:]
#     p = url.find(":")
#     parts["host"] = url[:p]
#     url = url[p+1:]
#     p = url.find("/")
#     parts["port"] = int(url[:p])
#     url = url[p+1:]
#     p = url.find("/")
#     if p == -1:
#         parts["database"] = url
#     else:
#         parts["database"] = url[:p]
#         parts["table"] = url[p+1:]
#     # end
#     if qargs.startswith("table="):
#         parts["table"] = qargs[6:]
#     if qargs.startswith("sql="):
#         for kv in qargs.split("&"):
#             k, v = kv.split("=")
#
#             # convert lists in tuples!
#             # this is necessary for 'SELECT ... FROM ... WHERE col IN :param ...'
#             if k == "sql":
#                 parts["sql"] = v
#             else:
#                 parts["params"][k]= v
#     # end
#     return parts


# def _parse_params(config: dict) -> dict:
#     params = config["params"]
#     for k in params.keys():
#         params[k] = _safe_val(params[k])
#     return params


# def _to_url_select(config):
#     url_config = dict_select(config, ['drivername','user', 'username', 'password', 'host', 'port', 'database'])
#     if 'user' in url_config:
#         url_config['username'] = url_config['user']
#         del url_config['user']
#     url: URL = URL.create(**url_config)
#
#     if "table" in config:
#         table = config["table"]
#         sql = f'select * from {table}'
#     else:
#         sql = config["sql"]
#
#     return url, sql


def _read_database(dburl: str, dtype, **kwargs):
    return read_sql(dburl, **kwargs)

    # config = _parse_url(dburl) | kwargs
    #
    # dburl, sql = _to_url_select(config)
    # params = _parse_params(config)
    #
    # # convert lists in tuples!
    # # this is necessary for 'SELECT ... FROM ... WHERE col IN :param ...'
    # engine = create_engine(dburl)
    # with engine.connect() as con:
    #     query = text(sql)
    #     df = read_sql(sql=query, con=con, params=params)
    # return df
# end


# ---------------------------------------------------------------------------
# read_data
# ---------------------------------------------------------------------------
# Extended version pf Pandas read_XXX to read files based on the file extension
# and to apply several transformations directly on the readed data.
#
# File types supported:
#
#   CSV
#   ARFF (used by Weka)
#   JSON
#   HTML
#   EXCEL
#   HDF
#   SPL table/query
#
# Transformations available:
#
#   conversion of datetime string into datetime object
#   conversion of categorical columns into
#       'category' Pandas data type
#       'boolean' Python value (supporting several way to represent boolean values)
#       'onehot' encoding, with automatic support for binary values (one single column)
#   conversion of integer columns into float
#   automatic creation fo the index based on one or multiple columns
#   dropping of NAN values
#   'ignore' columns (remove them from DF) or Pandas unnamed columns (with prefix 'Unnamed')
#   'rename' columns
#   'reindex' DF based on DF index
#

#
# dtype,
# categorical, boolean, numeric, onehot, binary,
# datetime, periodic,
# index, reindex,
# ignore, ignore_unnamed,
# count,
# na_values, dropna,
#

def read_data(
    file: str, *,
    dtype=None,             # list of columns types
    boolean=None,           # boolean columns
    integer=None,           # integer columns
    numeric=None,           # numerical/float columns
    categorical=None,       # pandas categorical columns

    onehot=None,            # categorical columns to convert using onehot encoding.
    binhot=None,            # categorical columns to convert using binary hot encoding.

    index=None,             # columns to use as index
    datetime_index=None,    # columns to use as datetime_index (alternative to 'index')
    ignore=None,            # columns to ignore
    ignore_unnamed=False,   # if to ignore 'Unnamed *' columns

    datetime=None,          # datetime column to convert in a PeriodTime

    periodic=None,          # [EXPERIMENTAL] to add datetime periodic representation
    count=False,            # [EXPERIMENTAL] if to add the column 'count' with value 1
    reindex=False,          # [EXPERIMENTAL] if to reindex the dataset
    sort=False,             # [EXPERIMENTAL] sort the data based on the index of the selected column(s)
    rename=None,            # [EXPERIMENTAL] rename some columns

    na_values=None,         # strings to consider NA values
    dropna=False,           # if to drop the rows containing NaN values
    fillna=None,            # if to fill na with a value
    **kwargs                # parameters to pass to 'pandas.read_*(...)' routine
) -> pd.DataFrame:
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
        na_values       scalar, str, list-like, or dict, optional
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
    :param boolean: column(s) to convert in 'boolean' value (False, True)
            It support several 'boolean' values:
                0/1, 'f'/'t', 'F'/'T', 'false'/'true', 'False'/'True', 'off'/'on', 'close'/'open'
    :param integer: column(s) to convert in 'int' type. To force int values on float columns
    :param numeric: column(s) to convert in 'float' type. To force float values on integer columns
    :param categorical: column(s) to convert in 'pandas.categorical' value
    :param onehot: column(s) to convert using onehot encoding
    :param binhot: column(s) to convert using binary hot encoding
    :param datetime: column used as datetime. Supported formats:
                - col: column already in datetime format
                - (col, format): 'format' used to convert the string into pandas datetime
                - (col, format, freq): 'freq' used to convert a datetime in a pandas period
    :param index: column(s) to use as pandas index.
            With multiple columns, it is created a MultiIndex following the columns order.
            Used to specify the index for a dataset multi-time series.
            The datetime index must be the last index in the list
            It is created a 'PriodIndex'
    :param datetime_index: as for 'index' but it is created a 'DatetimeIndex'
    :param count: if to add the column 'count' with value 1
    :param na_values: list of strings used as "NA" values
    :param dropna: if to drop rows containing NA values
            None/False:
                no row is removed
            True:
                rows containing NA values are removed
            ['col1', '...]:
                remove rows containing NA in the selected columns
    :param fillna: value to use to replace NA values.
           Note: it is an alternative to "dropna"
    :param periodic: if to add  one or multiple 'periodic' information (EXPERIMENTAL)

    :param reindex: if to 'reindex' the dataframe in such way to force ALL timestamp (EXPERIMENTAL)
    :param sort: resort the dataframe based on index or column(s) (EXPERIMENTAL)
    :param rename: rename some columns. The columns can be specified by position (starting from 0)
            or by name. The parameter can be a list (column specified inmplicitly by the position) or
            using a dictionary. As dictionary, the key if the original column name (an integer or a string)
            and the value, the new column name
            The column renaming is applied immediately after the reading.
    :param ignore_unnamed: if to ignore 'Unnamed: *' columns, created automatically by pandas read_*
            routines
    :param ignore: column(s) to ignore. The columns are removed as last step.
            This means that the columns are available for other processes before to remove them
            This means that the name to use must be consistent with the renaming.
    :param dict kwargs: extra parameters passed to pd.read_*()
    :return pd.DataFrame: a Pandas DataFrame
    """
    log = logging.getLogger("pandasx")

    # check parameter types
    assert isinstance(file, str), "'file' must be a str"

    # if 'dtype' is defined, compose the dictionary {col:dtype}
    if dtype is not None:
        h = _read_header(file)
        dt = _pandas_dtype(h, dtype)
    else:
        dt = None

    # move 'na_values' in kwargs
    if na_values is not None:
        kwargs['na_values'] = na_values

    # load the file based on extension
    # print(f"Loading {file} ...")
    log.info(f"Loading {file} ...")

    if file.endswith(".csv"):
        df = pd.read_csv(file, dtype=dt, **kwargs)
    elif file.endswith(".json"):
        # df = pd.read_json(file, dtype=dt, **kwargs)
        df = from_json(file, dtype=dt, **kwargs)
    elif file.endswith(".html"):
        df = pd.read_html(file, **kwargs)
    elif file.endswith(".xls"):
        df = pd.read_excel(file, dtype=dt, **kwargs)
    elif file.endswith(".xlsx"):
        df = pd.read_excel(file, dtype=dt, **kwargs)
    elif file.endswith(".hdf"):
        df = pd.read_hdf(file, dtype=dt, **kwargs)
    elif file.endswith(".arff"):  # extension
        df = read_arff(file, dtype=dt, **kwargs)
    elif "://" in file:  # extension
        df = _read_database(file, dtype=dt, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file}")

    return cleanup_data(
        df,
        dtype=dtype,
        boolean=boolean,
        integer=integer,
        numeric=numeric,
        categorical=categorical,
        onehot=onehot,
        binhot=binhot,
        index=index,
        datetime_index=datetime_index,
        ignore=ignore,
        ignore_unnamed=ignore_unnamed,
        datetime=datetime,
        periodic=periodic,
        count=count,
        reindex=reindex,
        sort=sort,
        rename=rename,
        dropna=dropna,
        fillna=fillna,
        # na_values=na_values,
        **kwargs
    )
# end


def cleanup_data(
        df: pd.DataFrame,
        dtype=None,         # list of columns types
        boolean=None,       # boolean columns
        integer=None,       # integer columns
        numeric=None,       # numerical/float columns

        categorical=None,   # pandas categorical columns
        onehot=None,        # categorical columns to convert using onehot encoding.
        binhot=None,        # categorical columns to convert using binary hot encoding.

        index=None,             # columns to use as index
        datetime_index=None,    # columns to use as datetime_index (alternative to 'index')
        ignore=None,            # columns to ignore
        ignore_unnamed=False,   # if to ignore 'Unnamed *' columns

        datetime=None,      # datetime column to convert in a PeriodTime

        periodic=None,      # [EXPERIMENTAL] to add datetime periodic representation
        count=False,        # [EXPERIMENTAL] if to add the column 'count' with value 1
        reindex=False,      # [EXPERIMENTAL] if to reindex the dataset
        sort=False,         # [EXPERIMENTAL] sort the data based on the index of the selected column(s)
        rename=None,        # [EXPERIMENTAL] rename some columns

        dropna=False,       # if to drop the rows containing NaN values
        fillna=None,        # if to fill NA values with an alternative value
        # na_values=None,     # strings to consider NaN values
        **kwargs            # parameters to pass to 'pandas.read_*(...)' routine
) -> pd.DataFrame:
    log = logging.getLogger("pandasx")

    assert isinstance(datetime, (NoneType, str, list, tuple)), \
        "'datetime' must be (None, str, (str, str), (str, str, str))"
    assert isinstance(periodic, (NoneType, bool, str, list, tuple)), \
        "'periodic' must be (None, str, (str, str), (str, dict))"
    assert isinstance(count, bool), \
        "'count' must be bool"
    assert isinstance(rename, (NoneType, list, tuple, dict)), \
        "'rename' must be (None, [str,...], {str:str...}"

    # convert list parameters passed as single value in a singleton list
    categorical = as_list(categorical, 'categorical')
    boolean = as_list(boolean, 'boolean')
    integer = as_list(integer, 'integer')
    numeric = as_list(numeric, 'numeric')
    onehot = as_list(onehot, 'onehot')
    binhot = as_list(binhot, 'binhot')
    ignore = as_list(ignore, 'ignore')
    index = as_list(index, 'index')
    datetime_index = as_list(datetime_index, 'datetime_index')

    # convert NaN in null
    df = df.replace({float("nan"): None})

    # parse the 'datetime' parameter
    datetime_name, datetime_format, datetime_freq = _parse_datetime(datetime)

    # select categorical/boolean columns
    if dtype is not None:
        categorical_, boolean_, ignore_ = _select_typed_columns(list(df.columns), dtype)
        categorical += categorical_
        boolean += boolean_
        ignore += ignore_
    # end

    # pandas categorical
    # for col in categorical:
    #     df[col] = df[col].astype('category')
    if categorical:
        df = type_encode(df, 'category', categorical)

    # pandas boolean
    # for col in boolean:
    #     df[col] = df[col].astype(bool)
    if boolean:
        df = type_encode(df, bool, boolean)

    # pandas integer
    # for col in integer:
    #     df[col] = df[col].astype(int)
    if integer:
        df = type_encode(df, int, integer)

    # pandas float
    # for col in numeric:
    #     df[col] = df[col].astype(float)
    if numeric:
        df = type_encode(df, float, numeric)

    # add the 'count' column
    # if count and 'count' not in df.columns:
    #     df['count'] = 1.
    if count:
        df = count_encode(df, count)

    # convert the datetime column
    if datetime is not None:
        df = datetime_encode(df, datetime)
        if datetime_freq is None:
            datetime_freq = infer_freq(df.index)

    # encode periodicity columns
    if periodic is not None:
        df = periodic_encode(df, periodic, datetime_name, datetime_freq)

    # onehot encoding
    if len(onehot) > 0:
        df = onehot_encode(df, onehot)

    # binhot encoding
    if len(binhot) > 0:
        df = binhot_encode(df, binhot)

    # add 'Unnamed: ...' columns to the list of columns to remove
    if ignore_unnamed:
        ignore += find_unnamed_columns(df)

    # remove the 'ignore' columns
    if len(ignore) > 0:
        df = columns_ignore(df, ignore)

    # remove the rows containing NaN
    # [extension] if dropna is 'list[str]'
    # note: it is better to check for NaN
    #       AFTER ignored columns, because some columns could be ignored because
    #             they have a lot of nan values!
    #       BEFORE to move the columns as index
    #       BEFORE to rename columns
    if dropna:
        df = nan_drop(df, columns=dropna)
    if fillna is not None:
        df = nan_fill(df, fillna=fillna)

    # force a sort
    # Note: here it is possible to use columns not yet removed
    if sort is not False:
        df = dataframe_sort(df, sort=sort)

    # compose the index
    if len(index) > 0:
        df = set_index(df, index, inplace=True)
    if len(datetime_index) > 0:
        df = set_index(df, datetime_index, inplace=True, as_datetime=True)

    # rename the columns
    if rename is not None:
        df = columns_rename(df, rename)

    # force the reindex
    if reindex:
        df = datetime_reindex(df)

    # print(f"... done {df.shape}")
    log.info(f"... done {df.shape}")
    return df
# end


# DataFrame.to_<type>()
#   to_orc(path=None, *, engine='pyarrow', index=None, engine_kwargs=None)
#   to_parquet(path=None, *, engine='auto', compression='snappy', index=None, partition_cols=None, storage_options=None, **kwargs)
#  =to_pickle(path, *, compression='infer', protocol=5, storage_options=None)
#  =to_csv(path_or_buf=None, *, sep=',', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression='infer', quoting=None, quotechar='"', lineterminator=None, chunksize=None, date_format=None, doublequote=True, escapechar=None, decimal='.', errors='strict', storage_options=None)
#  =to_hdf(path_or_buf, *, key, mode='a', complevel=None, complib=None, append=False, format=None, index=True, min_itemsize=None, nan_rep=None, dropna=None, data_columns=None, errors='strict', encoding='UTF-8')
#   to_sql(name, con, *, schema=None, if_exists='fail', index=True, index_label=None, chunksize=None, dtype=None, method=None)
#   to_dict(orient='dict', *, into=<class 'dict'>, index=True)
#  =to_excel(excel_writer, *, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, inf_rep='inf', freeze_panes=None, storage_options=None, engine_kwargs=None)
#  =to_json(path_or_buf=None, *, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=None, indent=None, storage_options=None, mode='w')
#  =to_xml(path_or_buffer=None, *, index=True, root_name='data', row_name='row', na_rep=None, attr_cols=None, elem_cols=None, namespaces=None, prefix=None, encoding='utf-8', xml_declaration=True, pretty_print=True, parser='lxml', stylesheet=None, compression='infer', storage_options=None)
#   to_html(buf=None, *, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', bold_rows=True, classes=None, escape=True, notebook=False, border=None, table_id=None, render_links=False, encoding=None)
#   to_feather(path, **kwargs)
#   to_latex(buf=None, *, columns=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None, decimal='.', multicolumn=None, multicolumn_format=None, multirow=None, caption=None, label=None, position=None)
#   to_stata(path, *, convert_dates=None, write_index=True, byteorder=None, time_stamp=None, data_label=None, variable_labels=None, version=114, convert_strl=None, compression='infer', storage_options=None, value_labels=None)
#   to_gbq(destination_table, *, project_id=None, chunksize=None, reauth=False, if_exists='fail', auth_local_webserver=True, table_schema=None, location=None, progress_bar=True, credentials=None)
#   to_records(index=True, column_dtypes=None, index_dtypes=None)
#   to_string(buf=None, *, columns=None, col_space=None, header=True, index=True, na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True, justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.', line_width=None, min_rows=None, max_colwidth=None, encoding=None)
#   to_clipboard(*, excel=True, sep=None, **kwargs)
#   to_markdown(buf=None, *, mode='wt', index=True, storage_options=None, **kwargs).
#

FORMAT_PARAMS = ["na_values", "datetime"]


def write_data(df: pd.DataFrame, file: Optional[str], **kwargs):
    ret = None

    # handle FORMAT_PARAMS
    if "datetime" in kwargs:
        df = _format_datetime(df, kwargs["datetime"])
    if "string" in kwargs:
        df = _format_strings(df, kwargs["string"])
    if "na_values" in kwargs:
        df = _format_na_values(df, kwargs["na_values"])

    # exclude ["na_values", "datetime"]
    kwargs = dict_exclude(kwargs, FORMAT_PARAMS)

    if file is None:
        ret = df.to_dict(**kwargs)
    elif file.endswith("csv"):
        if "index" not in kwargs:
            kwargs['index'] = False
        ret = df.to_csv(file, **kwargs)
    elif file.endswith(".json"):
        # df.to_json(file, **kwargs)
        ret = to_json(df, file, **kwargs)
    elif file.endswith(".xml"):
        ret = df.to_xml(file, **kwargs)
    elif file.endswith(".pickle"):
        df.to_pickle(file, **kwargs)
    elif file.endswith(".hdf5"):
        df.to_hdf(file, **kwargs)
    elif file.endswith(".xlsx"):
        df.to_excel(file, **kwargs)
    elif file.startswith("inline:") or file.startswith("memory:"):
        ret = write_dict(df, **kwargs)
    elif file.find("://") != -1:
        write_sql(df, file, **kwargs)
    else:
        raise ValueError(f"Unsupported file format {file}")
    return ret
# end


def _format_datetime(df: pd.DataFrame, datetime: Union[str, tuple[str, str]]) -> pd.DataFrame:
    from .json import _convert_datetime
    return _convert_datetime(df, datetime)
# end


def _format_na_values(df: pd.DataFrame, na_value) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)
    if na_value is None:
        return df
    if isinstance(na_value, list):
        na_value = na_value[0]
    return nan_fill(df, fillna=na_value)
# end


# def _format_columns(df: pd.DataFrame, kwargs) -> pd.DataFrame:
#     if "datetime" in kwargs:
#         from .json import _convert_datetime
#         datetime = kwargs["datetime"]
#         df = _convert_datetime(df, datetime)
#     if "na_values" in kwargs:
#         na_value =  kwargs["na_values"]
#         if isinstance(na_value, list):
#             na_value = na_value[0]
#         if na_value is not None:
#             df.fillna(na_value)
#     return df
# # end


def _format_strings(df: pd.DataFrame, columns: Union[None, str, list[str]]) -> pd.DataFrame:
    assert isinstance(df, pd.DataFrame)

    columns = as_list(columns, 'columns')
    if len(columns) == 0:
        return df

    for col in columns:
        ser = df[col]
        ser = ser.map(str)
        df[col] = ser
    # end
    return df
# end


def write_dict(df: pd.DataFrame, orient: Optional[str] = None, **kwargs) -> dict:
    return df.to_dict(orient=orient, **kwargs)


# ---------------------------------------------------------------------------
# Alternative functions: load/save
# ---------------------------------------------------------------------------

def load(file: str,
         *,
         dtype=None,
         boolean=None,
         integer=None,
         numeric=None,
         categorical=None,
         onehot=None,
         binhot=None,
         index=None,
         ignore=None,
         ignore_unnamed=False,
         datetime=None,
         periodic=None,
         count=False,
         reindex=False,
         sort=False,
         rename=None,
         dropna=False,
         na_values=None,
         **kwargs
 ) -> pd.DataFrame:
    return read_data(
        file,
        dtype=dtype,
        boolean=boolean,
        integer=integer,
        numeric=numeric,
        categorical=categorical,
        onehot=onehot,
        binhot=binhot,
        index=index,
        ignore=ignore,
        ignore_unnamed=ignore_unnamed,
        datetime=datetime,
        periodic=periodic,
        count=count,
        reindex=reindex,
        sort=sort,
        rename=rename,
        dropna=dropna,
        na_values=na_values,
        **kwargs
    )


def save(df: pd.DataFrame, file: str, **kwargs):
    write_data(df, file, **kwargs)

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
