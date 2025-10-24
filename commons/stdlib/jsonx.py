#
# Extensions to 'json' standard package:
#
#   1) to use 'load' and 'dump' directly with a file path
#
#   NO! It introduces a lot of problems
#   2) returning an 'stdlib.dict', a dictionary with a lot
#      of improvements useful for configurations
#
# Added support for Pandas dataframes
#   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
#
#   DataFrame.to_json(path_or_buf=None, *, orient=None, date_format=None, double_precision=10, force_ascii=True,
#       date_unit='ms', default_handler=None, lines=False, compression='infer', index=None, indent=None,
#       storage_options=None, mode='w')
#
# 'orient':
#       Series      {‘split’, ‘records’, ‘index’, ‘table’}.
#       Dataframe   {‘split’, ‘records’, ‘index’, ‘columns’, ‘values’, ‘table’}.
#
#

# [2025/09/13] add parameters
#   $<varname>
#   ${<varname>}
#

#
# json.dumps(obj) -> str
# json.dump(ooj, fp)
# jsonx.dump(obj, file_path)
#

from __future__ import annotations

import datetime as dt
import json
import os
from datetime import datetime
from typing import Optional, Union, cast

import numpy as np
import numpy.dtypes as npdt
import pandas as pd


OPEN_KWARGS = ['mode', 'buffering', 'encoding', 'errors', 'newline', 'closefd', 'opener']
PANDAS_KWARGS = ['orient', 'date_format', 'double_precision', 'force_ascii', 'date_unit', 'index']


MONTH_FORMAT = '%Y-%m'
DAY_FORMAT = '%Y-%m-%d'
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'


# ---------------------------------------------------------------------------
# Pandas support
# ---------------------------------------------------------------------------

def pandas_to_jsonx(
        data: Union[pd.Series, pd.DataFrame],
        path: Optional[str] = None,
        date_format=None,
        **kwargs) -> dict:

    if date_format not in [None, 'iso', 'epoch']:
        data = data.copy()
        if isinstance(data, pd.DataFrame):
            for c in data.columns:
                s = data[c]
                if pd.DatetimeTZDtype.is_dtype(s.dtype):
                    data[c] = s.dt.strftime(date_format)
                elif pd.PeriodDtype.is_dtype(s.dtype):
                    data[c] = s.dt.strftime(date_format)
                elif s.dtype.name == 'datetime64[ns]':
                    data[c] = s.dt.strftime(date_format)
                pass
            pass
        pass
    # end

    assert isinstance(data, (pd.DataFrame, pd.Series))
    if path is None:
        dtnow = datetime.now()
        timestamp = dtnow.strftime("%Y%m%d_%H%M%S_%f")
        json_file = f"tmp-{timestamp}.json"
    else:
        json_file = path

    with open(json_file, mode="w") as fp:
        data.to_json(fp, **kwargs)

    if path is None:
        with open(json_file, mode='r') as t:
            jdata = json.load(t)
        os.remove(json_file)
    else:
        jdata = None

    return jdata
# end


# ---------------------------------------------------------------------------
# Numpy support
# ---------------------------------------------------------------------------

NP_FLOAT_TYPES = (
    npdt.Float16DType,
    npdt.Float32DType,
    npdt.Float64DType,
    npdt.LongDoubleDType
)

NP_INT_TYPES = (
    npdt.Int8DType,
    npdt.Int16DType,
    npdt.Int32DType,
    npdt.Int64DType,

    npdt.UInt8DType,
    npdt.UInt16DType,
    npdt.UInt32DType,
    npdt.UInt64DType,

    npdt.ByteDType,
    npdt.ShortDType,
    npdt.IntDType,
    npdt.LongDType,
    npdt.LongLongDType,

    npdt.UByteDType,
    npdt.UShortDType,
    npdt.UIntDType,
    npdt.ULongDType,
    npdt.ULongLongDType,
)

NP_TIME_TYPES = (
    npdt.DateTime64DType,
    npdt.TimeDelta64DType
)


def ndarray_to_jsonx(a: np.ndarray) -> list[int|float|str]:
    atype = a.dtype
    if atype in NP_FLOAT_TYPES:
        return list(map(float, a))
    elif atype in NP_INT_TYPES:
        return list(map(int, a))
    elif atype in NP_TIME_TYPES:
        return list(map(lambda t: t.to_datetime().format(TIMESTAMP_FORMAT), a))
    else:
        return list(a)
# end


# ---------------------------------------------------------------------------
# JSONX encoder
# ---------------------------------------------------------------------------

def period_to_jsonx(period: pd.Period) -> str:
    return str(period)


def timestamp_to_jsonx(timestamp: pd.Timestamp) -> str:
    return str(timestamp)


JSONX_ENCODERS = {
    # Python datetime
    'datetime.datetime': lambda o: cast(dt.datetime, o).strftime(TIMESTAMP_FORMAT),
    'datetime.date': lambda o: cast(dt.date, o).strftime(DAY_FORMAT),

    # Numpy array
    'numpy.ndarray': ndarray_to_jsonx,
    'numpy.int64': int,
    'numpy.float16': float,
    'numpy.float32': float,
    'numpy.float64': float,

    # Pandas Series & DataFrame
    # 'pandas.core.series.Series': ser_to_jsonx,
    # 'pandas.core.frame.DataFrame': df_to_jsonx,
    'pandas._libs.tslibs.period.Period': period_to_jsonx,
    'pandas._libs.tslibs.timestamps.Timestamp': timestamp_to_jsonx,
    'pandas._libs.missing.NAType': (lambda o: None),
    # 'pandas.core.indexes.base.Index': lambda o: cast(pd.Index, o).tolist()
}


class JSONxEncoder(json.JSONEncoder):
    def __init__(self, **kwargs):
        self._pandas_kwargs = _dict_select(kwargs, PANDAS_KWARGS)
        kwargs = _dict_select(kwargs, PANDAS_KWARGS, exclude=True)
        super().__init__(**kwargs)

    def default(self, o):
        t = type(o)

        if t in [pd.Series, pd.DataFrame]:
            return pandas_to_jsonx(o, path=None, **self._pandas_kwargs)

        q = f'{t.__module__}.{t.__name__}'
        if q in JSONX_ENCODERS:
            return JSONX_ENCODERS[q](o)
        elif isinstance(o, tuple):
            return [
                self.encode(e)
                for e in o
            ]
        elif hasattr(o, 'to_json'):
            return o.to_json()
        else:
            return super().default(o)

    def encode(self, o):
        return super().encode(o)


def _dict_select(d:dict, keys: list[str], exclude=False) -> dict:
    if exclude:
        e = {}
        for k in d:
            if k not in keys:
                e[k] = d[k]
    else:
        e = {}
        for k in d:
            if k in keys:
                e[k] = d[k]
    return e


def _normalize(obj):
    if not isinstance(obj, dict):
        return obj
    odict = cast(dict, obj)
    if len(odict) == 0:
        return odict
    if not isinstance(list(odict.keys())[0], tuple):
        return odict

    def _set(d: dict, keys: tuple, v):
        n = len(keys)
        for i in range(n-1):
            k = keys[i]
            if k not in d:
                d[k] = {}
            d = d[k]
        k = keys[-1]
        d[k] = v
    # end

    ndict = {}
    for t, v in odict.items():
        _set(ndict, t, v)

    return ndict



def _fill_params(jdata: dict, params: dict):
    def _check(name):
        if name not in params:
            raise KeyError(f"Parameter '{name}' not specified")
    def _replace(v: str):
        if v.startswith("${") and v.endswith("}"):
            name = v[2:-1]
            _check(name)
            v = params[name]
        elif "${" in v:
            s = v.find("${")
            e = v.find("}", s)
            name = v[s+2:e]
            _check(name)
            v = v[:s] + str(params[name]) + v[e+1:]
        elif v.startswith("$"):
            name = v[1:]
            _check(name)
            v = params[name]
        return v

    for k in jdata:
        v = jdata[k]
        if isinstance(v, str):
            jdata[k] = _replace(v)
        elif isinstance(v, dict):
            _fill_params(v, params)
    return jdata


# ---------------------------------------------------------------------------
# jsonx.load(file) -> obj
# jsonx.dump(obj, file)
# ---------------------------------------------------------------------------

def load(file: str, **kwargs) -> dict:
    """
    Load the JSON in the specified file.
    It can replace strings in format:

        $<varname>
        ${varname}

    with the value passed in kwargs.

    :param file:
    :param kwargs:
    :return:
    """
    open_kwargs = _dict_select(kwargs, OPEN_KWARGS)
    with open(file, mode="r", **open_kwargs) as fp:
        jdata = dict(json.load(fp))
    return _fill_params(jdata, kwargs)


def loads(s, **kwargs) -> dict:
    jdata = dict(json.loads(s, **kwargs))
    return _fill_params(jdata, kwargs)


def dump(obj, file: str, **kwargs):
    if 'indent' not in kwargs:
        kwargs['indent'] = 4

    open_kwargs = _dict_select(kwargs, OPEN_KWARGS)
    obj = _normalize(obj)

    if isinstance(file, str):
        with open(file, mode="w", **open_kwargs) as fp:
            json.dump(obj, fp, cls=JSONxEncoder, **kwargs)
    else:
        json.dump(obj, file, cls=JSONxEncoder, **kwargs)
# end


def dumps(obj, *, skipkeys=False, ensure_ascii=True, check_circular=True,
        allow_nan=True, cls=None, indent=None, separators=None,
        default=None, sort_keys=False, **kw):
    return json.dumps(obj, skipkeys=skipkeys, ensure_ascii=ensure_ascii,
                      check_circular=check_circular, allow_nan=allow_nan,
                      cls=cls, indent=indent, separators=separators,
                      default=default, sort_keys=sort_keys, **kw)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
