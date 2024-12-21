#
# Note: now, 'jsonx' is able to serialize in JSON format
#       several 'almost-standard' data types:
#
#       Python data/datetime
#       Pandas Series, DataFrame
#       Numpy arrays
#
#
from __future__ import annotations

import os
from datetime import datetime
from typing import Literal, Optional, Union, Callable, Any, cast

import pandas as pd
from pandas._typing import (
    FilePath,
    ReadBuffer,
    WriteBuffer,
    TimeUnit,
    CompressionOptions,
    StorageOptions,
    JSONSerializable,
)
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from stdlib import dict_exclude
from stdlib import jsonx as json

# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

bool_t = bool


def _safeint(x):
    try:
        return int(x)
    except:
        return x
# end


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------

DF_TO_JSON_ORIENTS = ["split", "records", "index", "table", "columns", "values"]
ORIENTS_TYPE = Literal["split", "records", "index", "table", "columns", "values"]
ORIENTS_TYPEX = Literal["split", "records", "index", "table", "columns", "values", "list"]

def to_json(
        data: Union[pd.Series, pd.DataFrame],
        path_or_buf: FilePath | WriteBuffer[bytes] | WriteBuffer[str] | None = None,
        *,
        datetime: str | tuple[str, str] | None = None,
        orient: ORIENTS_TYPEX | None = None,
        date_format: str | None = None,
        double_precision: int = 10,
        force_ascii: bool_t = True,
        date_unit: TimeUnit = "ms",
        default_handler: Callable[[Any], JSONSerializable] | None = None,
        lines: bool_t = False,
        compression: CompressionOptions = "infer",
        index: bool_t | None = None,
        indent: int | None = None,
        storage_options: StorageOptions | None = None,
        mode: Literal["a", "w"] = "w",
    ) -> Optional[dict]:
    """
    Extended 'DataFrame.to_json(...)' to support:

        1) conversion in memory, returning a dict
        2) write on file
        3) the new data format 'list'

    :param data: DataFrame
    :param path_or_buf: path of the file, a buffer or None
    :param datetime: how to convert a datetime column
        If specified as string, ALL datetime/timestam columns are converted
        If specified as tuple (col, format), only the column is converted
    :param orient: JSON format to use
    :param date_format: date format to use
    :param double_precision: double precision to use
    :param force_ascii: force ascii
    :param date_unit: time unit to use
    :param default_handler: default handler
    :param lines: whether to write lines or not
    :param compression: compression option to use
    :param index: whether to index or not
    :param indent: indent size to use
    :param storage_options: storage options to use
    :param mode: mode to use
    :param kwargs: extra parameters passed to 'pandas.DataFrame.to_json(...)'
    :return: a dictionary, if in memory
    """
    assert isinstance(data, (pd.DataFrame, pd.Series))

    # exclude custom or explicit parameters
    kwargs = dict_exclude(locals(), ["data", "path_or_buf", "orient", "datetime"])
    jdata: Optional[dict] = None

    # special case: "orient='table'" requires "date_format='iso'"
    if orient == 'table':
        kwargs["date_format"] = 'iso'

    # convert dadetime column(s)
    data = _convert_datetime(data, datetime)

    if orient == "list":
        jdata = _to_flat_columns(data, path_or_buf, kwargs)
    elif orient in DF_TO_JSON_ORIENTS:
        jstr = data.to_json(path_or_buf, orient=cast(ORIENTS_TYPE, orient), **kwargs)
        if path_or_buf is None:
            jdata = json.loads(jstr)
    else:
        raise ValueError(f"Unsupported orient {orient}")

    return cast(Optional[dict], jdata)
# end

# ---------------------------------------------------------------------------

def _to_flat_columns(data: pd.DataFrame, json_file: str, kwargs: dict):
    jstr = data.to_json(orient="columns", **kwargs)
    jdata = json.loads(jstr)
    index = None
    for col in jdata.keys():
        if index is None:
            index = list(map(_safeint, jdata[col].keys()))
        values = list(jdata[col].values())
        jdata[col] = values
    # end
    jdata["$index"] = index

    if json_file is not None:
        with open(json_file, mode="w") as fp:
            json.dump(jdata, fp)
            jdata = None
    # end
    return jdata
# end


def _convert_datetime(df: pd.DataFrame, datetime: str | tuple[str,str] | None):
    if datetime is None:
        return df

    df = df.copy()

    if isinstance(datetime, (list, tuple)):
        dtcol = datetime[0]
        datetime = datetime[1]
    else:
        dtcol = None

    if dtcol is not None:
        df[dtcol] = df[dtcol].dt.strftime(datetime)
    else:
        for col in df.columns:
            if is_datetime(df[col]):
                df[col] = df[col].dt.strftime(datetime)
    return df
# end

# ---------------------------------------------------------------------------
# from_json
# ---------------------------------------------------------------------------

def from_json(
        path_or_buf: str | ReadBuffer[bytes] | ReadBuffer[str] | dict, *,
        orient: str | None = None,
        **kwargs
        # path_or_buf: FilePath | ReadBuffer[str] | ReadBuffer[bytes],
        # *,
        # orient: str | None = None,
        # typ: Literal["frame", "series"] = "frame",
        # dtype: DtypeArg | None = None,
        # convert_axes: bool | None = None,
        # convert_dates: bool | list[str] = True,
        # keep_default_dates: bool = True,
        # precise_float: bool = False,
        # date_unit: str | None = None,
        # encoding: str | None = None,
        # encoding_errors: str | None = "strict",
        # lines: bool = False,
        # chunksize: int | None = None,
        # compression: CompressionOptions = "infer",
        # nrows: int | None = None,
        # storage_options: StorageOptions | None = None,
        # dtype_backend: DtypeBackend | lib.NoDefault = lib.no_default,
        # engine: JSONEngine = "ujson",
) -> pd.DataFrame:
    """
    Extends 'pandas.read_json(...)' to support:

        1) conversion from memory, using a dict
        2) read from file
        3) the new data format 'list'

    :param path_or_buf: path of the file, a buffer or None
    :param orient: JSON format used
    :param kwargs: extra parameters passed to 'pandas.read_json(...)'
    :return: a pandas DataFrame
    """
    if isinstance(path_or_buf, dict):
        return _from_memory(path_or_buf, **kwargs)

    if orient is None and isinstance(path_or_buf, dict):
        orient = _guess_orient(path_or_buf)

    if orient == "list":
        return _from_flat_columns(path_or_buf, **kwargs)
    else:
        return pd.read_json(path_or_buf, orient=orient, **kwargs)

    # if orient == "list":
    #     return _from_flat_columns(path_or_buf)
    # return pd.read_json(
    #     path_or_buf=path_or_buf,
    #     orient=orient,
    #     typ=typ,
    #     dtype=dtype,
    #     convert_axes=convert_axes,
    #     convert_dates=convert_dates,
    #     keep_default_dates=keep_default_dates,
    #     precise_float=precise_float,
    #     date_unit=date_unit,
    #     encoding=encoding,
    #     encoding_errors=encoding_errors,
    #     lines=lines,
    #     chunksize=chunksize,
    #     compression=compression,
    #     nrows=nrows,
    #     storage_options=storage_options,
    #     dtype_backend=dtype_backend,
    #     engine=engine
    # )
# end

# ---------------------------------------------------------------------------

def _guess_orient(data: dict) -> str:
    assert isinstance(data, dict|list)
    if isinstance(data, list):
        return "records"
    if "columns" in data and "data" in data and "index" in data:
        return "split"
    if "shema" in data and"data" in data:
        return "table"
    if "$index" in data:
        return "list"
    raise ValueError("Unable to guess data 'format'")
# end


def _from_flat_columns(path: str, **kwargs) -> pd.DataFrame:
    jflat = json.load(path)
    jdata = {}
    index = jflat["$index"]
    n = len(index)
    for col in jflat.keys():
        if col == "$index": continue
        values = jflat[col]
        jdata[col] = {str(index[i]): values[i] for i in range(n)}

    dtnow = datetime.now()
    timestamp = dtnow.strftime("%Y%m%d_%H%M%S_%f")
    json_file = f"tmp-{timestamp}.json"
    json.dump(jdata, json_file)
    try:
        df = pd.read_json(json_file, orient="columns", **kwargs)
    finally:
        os.remove(json_file)
    return df
# end

# ---------------------------------------------------------------------------

def _from_memory(jdata: dict, **kwargs) -> pd.DataFrame:
    dtnow = datetime.now()
    timestamp = dtnow.strftime("%Y%m%d_%H%M%S_%f")
    json_file = f"tmp-{timestamp}.json"
    json.dump(jdata, json_file)

    try:
        orient = kwargs.get('orient', None)
        if orient == "list":
            df = _from_flat_columns(json_file, **kwargs)
        else:
            df = pd.read_json(json_file, orient=orient, **kwargs)
    finally:
        os.remove(json_file)
    return df
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
