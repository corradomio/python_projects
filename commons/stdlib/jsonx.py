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

# from .dict import dict

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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%ff')
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
        self._pandas_kwargs = {}
        for k in PANDAS_KWARGS:
            if k in kwargs:
                self._pandas_kwargs[k] = kwargs[k]
                del kwargs[k]
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


# ---------------------------------------------------------------------------
# jsonx.load(file) -> obj
# jsonx.dump(obj, file)
# ---------------------------------------------------------------------------

def load(file: str, **kwargs) -> dict:
    with open(file, mode="r", **kwargs) as fp:
        return dict(json.load(fp))


def loads(s, **kwargs) -> dict:
    return dict(json.loads(s, **kwargs))


def dump(obj, file: str, **kwargs):
    if 'indent' not in kwargs:
        kwargs['indent'] = 4

    open_kwargs = {}
    for k in OPEN_KWARGS:
        if k in kwargs:
            open_kwargs[k] = kwargs[k]
            del kwargs[k]

    if isinstance(file, str):
        with open(file, mode="w", **open_kwargs) as fp:
            json.dump(obj, fp, cls=JSONxEncoder, **kwargs)
    else:
        json.dump(obj, file, cls=JSONxEncoder, **kwargs)
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
