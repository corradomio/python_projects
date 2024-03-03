import math
from typing import Optional, Union, Collection

import pandas as pd


NAME_DATETIME = "$DT"


# ---------------------------------------------------------------------------
# set_datetime_index
# ---------------------------------------------------------------------------

def set_datetime_index(df: pd.DataFrame, datetime: Union[str, list[str]]) -> pd.DataFrame:
    """
    Assign to df the index extracted from the column 'datetime'

    :param df: dataframe
    :datetime: the formats are:
            str:            the column
            str, str:       the column, the format
            str, str, str:  the column, the format, the frequency
    """
    format = None
    freq = None
    if isinstance(datetime, str):
        pass
    elif len(datetime) == 2:
        datetime, format = datetime
    elif len(datetime) == 3:
        datetime, format, freq = datetime

    dt = df[datetime]
    if format is not None:
        dt = pd.to_datetime(dt, format=format)
    if freq is not None:
        dt = dt.dt.to_period(freq)

    df = df.set_index(dt)
    return df
# end


# ---------------------------------------------------------------------------
# periodic_encode
# ---------------------------------------------------------------------------

def periodic_encode(df: pd.DataFrame, periodic, datetime_name, datetime_freq):
    if not isinstance(periodic, Collection):
        periodic = [periodic]

    for p in periodic:
        if p in [False, None, 'none', '']:
            pass
        elif p == 'last_week_in_month':
            df = _last_week_in_month(df, datetime_name)
        elif isinstance(p, dict):
            df = _periodic_encode(df, datetime_name, freq=datetime_freq, **p)
        elif isinstance(p, str):
            df = _periodic_encode(df, datetime_name, p, freq=datetime_freq)
        elif isinstance(p, (list, tuple)):
            df = _periodic_encode(df, datetime_name, *p, freq=datetime_freq)
        else:
            df = _periodic_encode(df, datetime_name, p, freq=datetime_freq)

    return df
# end


def _last_week_in_month(df: pd.DataFrame, datetime: Optional[str] = None):
    """
    Add the column '<datetime>_lweek' containing the value 0,1 if the day
    is located in the last week of the month

    :param df: dataframe to process
    :param datetime: if to use a datetime column. If None, it is used the index
    :return:
    """
    def lweek(t) -> int:
        if isinstance(t, pd.Period):
            # Period is composed by 2 timestamps:
            #
            #   start_time
            #   end_time
            #
            # The properties (day, day_of_week, ...) are refereed to 'end_time'
            #

            # check if t.day is in the last week of the month
            # if t.day//7 < 4, it is NOT the last week of the month
            if t.day//7 < 4:
                return int(False)

            # Problem: it is necessary to understand which is the 'periodicity':
            # - daily
            # - weekly
            if isinstance(t.freq, pd.tseries.offsets.Week):
                return int(t.month != (t+1).month)
            elif isinstance(t.freq, pd.tseries.offsets.Day):
                return int(t.month != (t+7).month)
            else:
                raise ValueError(f"Unsupported frequency '{t.freqstr}'")
        elif isinstance(t, pd.DatetimeTZDtype):
            raise ValueError(f"Unsupported datetime of type '{t.__class__.__name__}'")
        else:
            raise ValueError(f"Unsupported datetime of type '{t.__class__.__name__}'")
        pass

    if datetime is None:
        dt = df.index.to_series()
        if dt.name is None:
            lweek_name = f"{NAME_DATETIME}_lweek"
        else:
            lweek_name = f"{dt.name}_lweek"
    else:
        lweek_name = f"{datetime}_lweek"
        dt = df[datetime]

    dt = dt.apply(lweek)
    df[lweek_name] = dt
    return df


def _periodic_encode(df: pd.DataFrame,
                     datetime: Optional[str] = None,
                     method: str = 'onehot',
                     freq: Optional[str] = None,
                     year_scale=None,
                     columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Add some extra column to represent a periodic time

    Supported methods:

        onehot:     year/month          -> year, month (onehot encoded)
        order:      year/month          -> year, month (index, 0-based)
                    year/month/day      -> year, month (index, 0-based), day (0-based)
        circle      year/month/day/h    -> year, month, day, cos(h/24), sin(h/24)
                    year/month/day      -> year, month, cos(day/30), sin(day/30)
                    year/month          -> year, cos(month/12), sin(month/12)
        sincos|cossin: as circle

    :param df: dataframe to process
    :param datetime: if to use a datetime column. If None, it is used the index
    :param method: method to use
    :param columns: column names to use. The n of columns depends on the method
    :param year_scale: values used to scale the years
            It can be:
                1. None
                    no scale is applied
                2. (y0, y1)
                    the year y0 is scaled to 0
                    the year y1 is scaled to 1
                3. (y0, s0, y1, s1)
                    the year y0 is scaled to to s0
                    the year y1 is scaled to to s1

    :param freq: frequency ('H', 'D', 'W', 'M')
    :return:
    """
    if method in [None, '']:
        pass
    elif method in ['circle', 'sincos', 'cossin']:
        df = _sincos_encoder(df, datetime, columns, year_scale, freq)
    elif method == 'onehot':
        df = _onehot_encode(df, datetime, columns, year_scale, freq)
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


def _scale_year(year, year_scale):
    if year_scale is None:
        return year
    if len(year_scale) == 2:
        y0, y1 = year_scale
        s0, s1 = 0, 1
    else:
        y0, s0, y1, s1 = year_scale

    dy = y1 - y0
    ds = s1 - s0

    year = year.apply(lambda y: s0 + (y - y0) * ds / dy)
    return year


def _onehot_encode(df, datetime, columns, year_scale, freq):
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

    # dty = _scale_year(dty, year_scale)
    # df[columns[0]] = dty

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


def _sincos_encoder(df, datetime, columns, year_scale, freq):
    if datetime is None:
        dt = df.index.to_series()
        index_name = df.index.name
        datetime = index_name if index_name else "dt"
    else:
        dt = df[datetime]

    columns = _columns_name(columns, datetime, ["_c", "_s"])

    if freq == 'M':
        FREQ = 2 * math.pi / 12
        dtcos = dt.apply(lambda x: math.cos(FREQ * (x.month - 1)))
        dtsin = dt.apply(lambda x: math.sin(FREQ * (x.month - 1)))
    elif freq == 'W':
        FREQ = 2 * math.pi / 7
        dtcos = dt.apply(lambda x: math.cos(FREQ * (x.weekday)))
        dtsin = dt.apply(lambda x: math.sin(FREQ * (x.weekday)))
    elif freq == 'D':
        FREQ = 2 * math.pi / 24
        dtcos = dt.apply(lambda x: math.cos(FREQ * (x.hour)))
        dtsin = dt.apply(lambda x: math.sin(FREQ * (x.hour)))
    else:
        raise ValueError(f"Unsupported frequency '{freq}'")

    df[columns[0]] = dtcos
    df[columns[1]] = dtsin

    return df


last_week_in_month = _last_week_in_month

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
