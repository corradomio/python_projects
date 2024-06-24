import pandas as pd
from typing import Iterator
from pandas._libs import lib
from pandas import DataFrame


def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
    dtype_backend=lib.no_default,
    dtype=None,
) -> DataFrame | Iterator[DataFrame]:
    """
    It extends 'pandas.read_sql' forcing all 'Not a Number' values to be 'pd.NA'

    Note: pd.NA is an experimental value. Why it is used this!
    """
    df = pd.read_sql(
        sql, con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        columns=columns,
        chunksize=chunksize,
        dtype_backend=dtype_backend,
        dtype=dtype
    )
    df.fillna(pd.NA, inplace=True)
    return df


def read_sql_query(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    chunksize=None,
    dtype=None,
    dtype_backend=lib.no_default,
) -> DataFrame | Iterator[DataFrame]:
    """
    It extends 'pandas.read_sql_query' forcing all 'Not a Number' values to be 'pd.NA'

    Note: pd.NA is an experimental value. Why it is used this!
    """

    df = pd.read_sql_query(
        sql, con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        chunksize=chunksize,
        dtype=dtype,
        dtype_backend=dtype_backend,
    )
    df.fillna(pd.NA, inplace=True)
    return df