from __future__ import annotations

import json
import os
import numpy.dtypes as npdt
from datetime import datetime
from typing import (Literal, Optional)
from typing import Union

from numpy.dtypes import *
import pandas as pd


DATEFORMAT = "dateformat"
DEFAULT_DATEFORMAT = "%Y-%m-%d %H:%M:%S"
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

ORIENT_VALUES = ['split', 'records', 'index', 'columns', 'values', 'table']


def _idxtoj(idx: pd.Index) -> list[Union[str, float, int, bool, None]]:
    #   RangeIndex
    #   MultiIndex
    #   IntervalIndex
    #   DatetimeIndex
    #   CategoricalIndex
    #   PeriodIndex
    if isinstance(idx, pd.RangeIndex):
        return list(map(int, idx.values))
    elif isinstance(idx, pd.DatetimeIndex):
        return list(map(lambda t: t.format("%Y/%m/%d %H:%M:%S"), idx.values))
    elif isinstance(idx, pd.PeriodIndex):
        return list(map(lambda t: t.format("%Y/%m/%d %H:%M:%S"), idx.values))
    elif isinstance(idx, pd.Index):
        return list(map(int, idx.values))
    else:
        raise ValueError(f"Unsupported index {type(idx)}")
# end


def _stoj(ser: Union[pd.Index, pd.Series]) -> list[Union[str, float, int, bool, None]]:
    if isinstance(ser.dtype, DateTime64DType):
        # return list(map(lambda t: t.to_datetime().format("%Y/%m/%d %H:%M:%S"), ser.values))
        return list(map(str, ser.values))
    elif isinstance(ser.dtype, TimeDelta64DType):
        return list(map(lambda t: t.to_datetime().format("%Y/%m/%d %H:%M:%S"), ser.values))
    elif isinstance(ser.dtype, (Float16DType, Float32DType, Float64DType, LongDoubleDType)):
        return list(map(lambda t: float(t), ser.values))
    elif isinstance(ser.dtype, (Int8DType, Int16DType, Int64DType)):
        return list(map(lambda t: int(t), ser.values))
    elif isinstance(ser.dtype, (ByteDType, ShortDType, IntDType, LongDType, LongLongDType)):
        return list(map(lambda t: int(t), ser.values))
    elif isinstance(ser.dtype, (UInt8DType, UInt16DType, UInt64DType)):
        return list(map(lambda t: int(t), ser.values))
    elif isinstance(ser.dtype, (UByteDType, UShortDType, UIntDType, ULongDType, ULongLongDType)):
        return list(map(lambda t: int(t), ser.values))
    else:
        return list(ser.values)
# end


def to_json(
    data: Union[pd.Series, pd.DataFrame],
    path: Optional[str] = None,
    orient: Optional[Literal["split", "records", "index", "table", "columns", "values"]] = None,
    **kwargs) -> dict:

    assert isinstance(data, (pd.DataFrame, pd.Series))

    if orient in ORIENT_VALUES:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tmp = f"tmp-{timestamp}.json"
        data.to_json(tmp, orient=orient, **kwargs)
        with open(tmp, mode='r') as t:
            jdata = json.load(t)
        os.remove(tmp)
        return jdata
    elif isinstance(data, pd.Series):
        jdata = {}
        # orient: ‘split’, ‘records’, ‘index’, ‘table’}
        jdata['$index'] = _idxtoj(data.index)
        jdata[data.name] = _stoj(data)
    else:
        jdata = {}
        # orient: {‘split’, ‘records’, ‘index’, ‘columns’, ‘values’, ‘table’}
        for col in data.columns:
            jdata[col] = _stoj(data[col])
        jdata['$index'] = _idxtoj(data.index)

    if path is not None:
        with open(path, mode='w') as fout:
            json.dump(jdata, fout, **kwargs)
    return jdata
# end
