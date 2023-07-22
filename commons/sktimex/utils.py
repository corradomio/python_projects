import math
from typing import Union, Optional, Any
import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCIKIT_NAMESPACES = ['sklearn', 'catboost', 'lightgbm', 'xgboost']
SKTIME_NAMESPACES = ['sktime']

FH_TYPES = Union[type(None), int, list[int], np.ndarray, ForecastingHorizon]
PD_TYPES = Union[type(None), pd.Series, pd.DataFrame]


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def import_from(qname: str) -> Any:
    import importlib
    p = qname.rfind('.')
    qmodule = qname[:p]
    name = qname[p+1:]

    module = importlib.import_module(qmodule)
    clazz = getattr(module, name)
    return clazz


def dict_del(d: dict, keys: Union[str, list[str]]) -> dict:
    d = {} | d
    if isinstance(keys, str):
        keys = list[keys]
    for k in keys:
        if k in d:
            del d[k]
    return d


def kwval(kwargs: dict[Union[str, tuple], Any], key: Union[str, tuple], defval: Any = None) -> Any:
    if key not in kwargs:
        return defval

    val = kwargs[key]
    if not isinstance(defval, str) and isinstance(val, str):
        if defval is None:
            return val
        if isinstance(defval, bool):
            return tobool(val)
        if isinstance(defval, int):
            return int(val)
        if isinstance(defval, float):
            return float(val)
        else:
            raise ValueError(f"Unsupported conversion from str to '{type(defval)}'")
    return val


# ---------------------------------------------------------------------------
# periodic_encode
# ---------------------------------------------------------------------------

def periodic_encode(df,
                    datetime: Optional[str] = None,
                    method: str = 'onehot',
                    columns: Optional[list[str]] = None,
                    freq: Optional[str] = None,
                    year_scale=None) -> pd.DataFrame:
    """
    Add some extra column to represent a periodic time

    Supported methods:

        onehot:     year/month      -> year, month (onehot encoded)
        order:      year/month      -> year, month (index, 0-based)
                    year/month/day  -> year, month (index, 0-based), day (0-based) 
        circle      year/month/day/h    year, month, day, cos(h/24), sin(h/24)
                    year/month/day      year, month, cos(day/30), sin(day/30)
                    year/month          year, cos(month/12), sin(month/12)

    :param df: dataframe to process
    :param datetime: if to use a datetime column. If None, it is used the index
    :param method: method to use
    :param columns: column names to use. The n of columns depends on the method
    :param freq: frequency ('H', 'D', 'W', 'M')
    :return:
    """
    if method == 'onehot':
        df = _onehot_encode(df, datetime, columns, freq, year_scale)
    elif method == 'order' and freq == 'M':
        df = _order_month_encoder(df, datetime, columns, year_scale)
    elif method == 'order' and freq == 'D':
        df = _order_day_encoder(df, datetime, columns, year_scale)
    elif method == 'M' or freq == 'M':
        df = _monthly_encoder(df, datetime, columns, year_scale)
    elif method == 'W' or freq == 'W':
        df = _weekly_encoder(df, datetime, columns, year_scale)
    elif method == 'D' or freq == 'D':
        df = _daily_encoder(df, datetime, columns, year_scale)
    else:
        raise ValueError(f"'Unsupported periodic_encode method '{method}/{freq}'")

    return df
# end


def _columns_name(columns, datetime, suffixes):
    if columns is None:
        columns = [datetime + s for s in suffixes]
    assert len(columns) == len(suffixes), f"It is necessary to have {len(suffixes)} columns ({suffixes})"
    return columns


def _scale_year(year, year_scaler):
    if year_scaler is None:
        return year

    if len(year_scaler) == 2:
        y1, s1 = year_scaler
        s0 = 0.
        y0 = year[0]
    else:
        y0, s0, y1, s1 = year_scaler

    dy = y1 - y0
    ds = s1 - s0

    year = year.apply(lambda y: s0 + (y - y0) * ds / dy)
    return year


def _onehot_encode(df, datetime, columns, freq, year_scale):
    if datetime is None:
        dt = df.index.to_series()
        datetime = "dt"
    else:
        dt = df[datetime]

    columns = _columns_name(
        columns, datetime,
        ["_y", "_01", "_02", "_03", "_04", "_05", "_06", "_07", "_08", "_09", "_10", "_11", "_12"]
        # "_y", "_jan", "_feb", "_mar", "_apr", "_may", "_jun", "_jul", "_aug", "_sep", "_oct", "_nov", "_dec"]
    )

    # year == 0
    dty = dt.apply(lambda x: x.year)
    dty = _scale_year(dty, year_scale)

    df[columns[0]] = dty

    # month  in range [1, 12]
    for month in range(1, 13):
        dtm = dt.apply(lambda x: int(x.month == month))
        df[columns[month]] = dtm

    return df


def _order_month_encoder(df, datetime, columns, year_scale):
    if datetime is None:
        dt = df.index.to_series()
        datetime = "dt"
    else:
        dt = df[datetime]

    columns = _columns_name(columns, datetime, ["_y", "_m"])

    dty = dt.apply(lambda x: x.year)
    dty = _scale_year(dty, year_scale)
    dtm = dt.apply(lambda x: x.month - 1)

    df[columns[0]] = dty
    df[columns[1]] = dtm

    return df


def _order_day_encoder(df, datetime, columns, year_scale):
    if datetime is None:
        dt = df.index.to_series()
        datetime = "dt"
    else:
        dt = df[datetime]

    columns = _columns_name(columns, datetime, ["_y", "_m", "_d"])

    dty = dt.apply(lambda x: x.year)
    dty = _scale_year(dty, year_scale)
    dtm = dt.apply(lambda x: x.month - 1)
    dtd = dt.apply(lambda x: x.day - 1)

    df[columns[0]] = dty
    df[columns[1]] = dtm
    df[columns[2]] = dtd

    return df


def _monthly_encoder(df, datetime, columns, year_scale):
    if datetime is None:
        dt = df.index.to_series()
        datetime = "dt"
    else:
        dt = df[datetime]
    FREQ = 2 * math.pi / 12

    columns = _columns_name(columns, datetime, ["_y", "_c", "_s"])

    dty = dt.apply(lambda x: x.year)
    dty = _scale_year(dty, year_scale)
    dtcos = dt.apply(lambda x: math.cos(FREQ * (x.month - 1)))
    dtsin = dt.apply(lambda x: math.sin(FREQ * (x.month - 1)))

    df[columns[0]] = dty
    df[columns[1]] = dtcos
    df[columns[2]] = dtsin

    return df


def _weekly_encoder(df, datetime, columns, year_scale):
    if datetime is None:
        dt = df.index.to_series()
        datetime = "dt"
    else:
        dt = df[datetime]
    FREQ = 2 * math.pi / 7

    columns = _columns_name(columns, datetime, ["_y", "_m", "_c", "_s"])

    dty = dt.apply(lambda x: x.year)
    dty = _scale_year(dty, year_scale)
    dtm = dt.apply(lambda x: x.month - 1)
    dtcos = dt.apply(lambda x: math.cos(FREQ * (x.weekday)))
    dtsin = dt.apply(lambda x: math.sin(FREQ * (x.weekday)))

    df[columns[0]] = dty
    df[columns[1]] = dtm
    df[columns[2]] = dtcos
    df[columns[3]] = dtsin

    return df


def _daily_encoder(df, datetime, columns, year_scale):
    if datetime is None:
        dt = df.index.to_series()
        datetime = "dt"
    else:
        dt = df[datetime]
    FREQ = 2 * math.pi / 24

    columns = _columns_name(columns, datetime, ["_y", "_m", "_d", "_c", "_s"])

    dty = dt.apply(lambda x: x.year)
    dty = _scale_year(dty, year_scale)
    dtm = dt.apply(lambda x: x.month - 1)
    dtd = dt.apply(lambda x: x.day - 1)
    dtcos = dt.apply(lambda x: math.cos(FREQ * (x.hour)))
    dtsin = dt.apply(lambda x: math.sin(FREQ * (x.hour)))

    df[columns[0]] = dty
    df[columns[1]] = dtm
    df[columns[2]] = dtd
    df[columns[4]] = dtcos
    df[columns[5]] = dtsin

    return df


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
