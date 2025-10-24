from typing import Any, Literal, Optional, Union
from datetime import datetime
import numpy as np
import numpy.dtypes as npt
import pandas as pd
import pandasx as pdx

from stdlib import is_instance
from stdlib.dateutilx import relativeperiods, relativedifference


# ---------------------------------------------------------------------------
# ensure_str_column_names
# ---------------------------------------------------------------------------

def ensure_str_column_names(df: pd.DataFrame, inplace=False) -> tuple[pd.DataFrame, dict[Any, str]]:
    # ensure all column names in the dataframe are strings
    to_rename: dict[Any, str] = {}
    for col in df.columns:
        if not isinstance(col, str):
            to_rename[col] = str(col)

    if len(to_rename) == 0:
        return df, to_rename

    df_upd = df.rename(columns=to_rename, inplace=inplace)
    return (df if inplace else df_upd), to_rename
# end


def ensure_orig_column_names(df: pd.DataFrame, renamed: dict[Any, str], inplace=False) -> pd.DataFrame:
    # invert the renaming
    if len(renamed) == 0:
        return df

    reversed = {renamed[k]: k for k in renamed}
    df_upd = df.rename(columns=reversed, inplace=inplace)
    return (df if inplace else df_upd)
# end


def map_column_names(df: pd.DataFrame, to_replace: dict, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    df.rename(columns=to_replace, inplace=True)
    return df


def map_column_values(df: pd.DataFrame, column: str, to_replace: dict, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()
    df[column].replace(to_replace=to_replace, inplace=True)
    return df


# ---------------------------------------------------------------------------
# set_nan_values
# ---------------------------------------------------------------------------

def set_nan_values(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    target_dict: dict[int, str],
    inplace=False
) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    new_format = 'date' in df.columns
    date_col = 'date' if new_format else "state_date"

    for measure_id in target_dict:
        mname = target_dict[measure_id] if new_format else measure_id
        df.loc[(df[date_col] >= start_date) & (df[date_col] <= end_date), mname] = pd.NA

    return df
# end


# ---------------------------------------------------------------------------
# set_consistent_index
# ---------------------------------------------------------------------------

def _set_consistent_datetime_column(df, freq, dtcol, position, start_date=None):
    n = len(df)
    begin_date: pd.Timestamp = df[dtcol].iloc[0]
    end_date: pd.Timestamp = df[dtcol].iloc[-1]
    # rp1 = relativeperiods(periods=1, freq=freq)

    if start_date is None:
        if position == "first":
            start_date = begin_date
        else:
            start_date = end_date
    # end

    if start_date <= begin_date:
        periods = pdx.relativedifference(start_date, end_date, freq=freq)
        last_date = start_date + relativeperiods(periods, freq)
        dr = pd.date_range(start_date, last_date)
        if end_date != dr[-1] or begin_date != dr[-n-1]:
            df[dtcol] = dr[-n:]
        pass
    elif start_date >= end_date:
        periods = pdx.relativedifference(begin_date, start_date, freq=freq)
        first_date = start_date + relativeperiods(periods, freq)
        dr = pd.date_range(first_date, start_date)
        if begin_date != dr[0] or end_date != dr[-1]:
            df[dtcol] = dr[:n]
        pass
    else:
        raise ValueError("Invalid start_date, it must be < df begin_date or > df end_date")

    return df


def set_consistent_datetime(
        df: pd.DataFrame, freq: Literal['D', 'W', 'M'],
        *,
        dtcol: Optional[str],
        position: Literal["first", "last"]="last",
        start_date: Optional[datetime] = None,
        inplace=False
) -> pd.DataFrame:
    #
    # WARNING: sometimes 'df' contains an inconsistent time index, that is,
    # it HAS the correct frequency BUT it DOESN'T start at the correct timestamp
    # This method consider a reference timestamp (the first or the last, or it is passed as parameter)
    # and it recreates the index consistent with the frequency
    #
    # WARNING: THIS IS A TRICK to avoid the problems in iPlan!
    #          IT HAS NO SENSE to consider the data on the Sunday also valid in Monday!!!!!
    #
    assert is_instance(df, pd.DataFrame) and len(df) > 0
    assert is_instance(dtcol, Optional[str])
    assert is_instance(position, Literal["first", "last"])
    assert is_instance(start_date, Optional[datetime])
    assert is_instance(freq, Literal['D', 'W', 'M'])

    if not inplace:
        df = df.copy()

    if dtcol is not None:
        return _set_consistent_datetime_column(df, freq, dtcol, position, start_date)
    else:
        raise NotImplemented("set_consistent_datetime on DataFrame.index")
# end


# ---------------------------------------------------------------------------
# categorical_columns
# ---------------------------------------------------------------------------
# Note: DONT change into a set {...} !!!!!

NUMPY_STRING_DTYPES = [
    "object", "str",
    # npt.ObjectDType,
    # npt.BytesDType,
    # npt.StrDType
]

NUMPY_NUMERICAL_DTYPES = [
    "int", "float", "bool",
    "float64", "float32", "float16",
    "int64", "int32", "int16", "int8",
    "uint64", "uint32", "uint16", "uint8"
    # npt.BoolDType,
    # npt.Int8DType,
    # npt.UInt8DType,
    # npt.Int16DType,
    # npt.UInt16DType,
    # npt.Int32DType,
    # npt.UInt32DType,
    # npt.Int64DType,
    # npt.UInt64DType,
    # npt.ByteDType,
    # npt.UByteDType,
    # npt.ShortDType,
    # npt.UShortDType,
    # npt.IntDType,
    # npt.UIntDType,
    # npt.LongDType,
    # npt.ULongDType,
    # npt.LongLongDType,
    # npt.ULongLongDType,
    # npt.Float16DType,
    # npt.Float32DType,
    # npt.Float64DType,
    # npt.LongDoubleDType
]


def categorical_columns(df: pd.DataFrame, excluding: Union[None, list[str], set[str]]=None) -> list[str]:
    if excluding is None:
        excluding = []

    cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in NUMPY_STRING_DTYPES:
            if col not in excluding:
                cols.append(col)
    return cols
# end

def numerical_columns(df: pd.DataFrame, excluding: Union[None, list[str], set[str]]=None) -> list[str]:
    if excluding is None:
        excluding = []

    cols = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype in NUMPY_NUMERICAL_DTYPES:
            if col not in excluding:
                cols.append(col)
    return cols
# end
