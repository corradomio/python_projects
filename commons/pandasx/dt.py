from typing import Literal, Union

import numpy as np
import pandas as pd
from datetime import datetime, date
from stdlib.dateutilx import relativeperiods


# ---------------------------------------------------------------------------
# to_date_type
# ---------------------------------------------------------------------------

def to_date_type(dt: Union[date, datetime], date_type):
    dt_type = type(dt)
    if dt_type == date_type:
        return dt
    if date_type == pd.Timestamp:
        if dt_type == date:
            return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day)
        if dt_type == datetime:
            return pd.Timestamp(year=dt.year, month=dt.month, day=dt.day, hour=dt.hour, minute=dt.minute, second=dt.second)
    return dt


# ---------------------------------------------------------------------------
# to_datetime
# ---------------------------------------------------------------------------

NP_FIRST_JAN_1970 = np.datetime64('1970-01-01T00:00:00')
NP_ONE_SECOND = np.timedelta64(1, 's')


def to_datetime(dt) -> datetime:
    """
    Try to convert into a datetime an object represented
    in several formats

    :param dt:  object to convert
    :return: datetime
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        return pd.to_datetime(dt).to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, pd.Period):
        return dt.to_timestamp().to_pydatetime()
    if isinstance(dt, np.datetime64):
        ts = ((dt - NP_FIRST_JAN_1970) / NP_ONE_SECOND)
        dt = datetime.utcfromtimestamp(ts)
        return dt
    else:
        raise ValueError(f"Unsupported datetime {dt}")


# ---------------------------------------------------------------------------
# to_period
# ---------------------------------------------------------------------------

def to_period(dt: datetime, freq=None) -> pd.Period:
    return pd.Period(dt, freq=freq)


# ---------------------------------------------------------------------------
# date_range
# ---------------------------------------------------------------------------

INCLUSIVE_TYPE = Literal['left', 'right', 'both', 'neither']


def _sdp(start_date: datetime, periods: int, delta) -> list[datetime]:

    dr = []
    cdt = start_date
    for p in range(periods):
        dr.append(cdt)
        cdt += delta

    return dr


def _edp(end_date: datetime, periods: int, delta) -> list[datetime]:

    dr = []
    cdt = end_date
    for p in range(periods):
        dr.append(cdt)
        cdt -= delta

    dr = dr[::-1]
    return dr


def _sedl(start_date, end_date, delta) -> list[datetime]:
    # start_date, end_date, left

    dr = []
    cdt = start_date
    ldt = start_date
    while cdt <= end_date:
        dr.append(cdt)
        ldt = cdt
        cdt += delta
    if ldt != end_date:
        dr.append(cdt)

    return dr


def _sedr(start_date, end_date, delta) -> list[datetime]:
    # start_date, end_date, right

    dr = []
    cdt = end_date
    ldt = end_date
    while cdt >= start_date:
        dr.append(cdt)
        ldt = cdt
        cdt -= delta
    if ldt != start_date:
        dr.append(cdt)

    dr = dr[::-1]
    return dr


def date_range(start=None, end=None, periods=None,
               freq=None, delta=None,
               name=None,
               inclusive: INCLUSIVE_TYPE = 'both', align='left') -> pd.DatetimeIndex:
    """
    This function has a behavior very similar than 'pd.date_range', with the following difference:
    it is possible to specify timestamp with a specified freq)ency, starting at each datetime.
    For example, it is possible to generate weekly timestamps starting from Wednesday, not from
    Sunday or Monday.

    It is also possible to generate sequences of timestamps starting from 'start_date' or 'end_date'
    Possibilities:

        start_date  periods     freq            inclusive       (forward)
        end_date    periods     freq            inclusive       (backward)
        start_date  end_date    freq    align   inclusive       (left:forward, right:backward)


    :param start:
    :param end:
    :param periods:
    :param freq:
    :param delta:
    :param name:
    :param inclusive:
    :param align:
    :return:
    """
    assert isinstance(freq, (type(None), str))
    assert delta is not None or freq is not None

    start_date = to_datetime(start)
    end_date = to_datetime(end)

    if delta is None:
        delta = relativeperiods(periods=1, freq=freq)

    if periods is not None:
        if inclusive in ['neither', 'right']:
            periods += 1
        if inclusive in ['neither', 'left']:
            periods += 1
    elif start_date is not None and end_date is not None:
        if inclusive in ['neither', 'right']:
            # start_date += delta
            pass
        if inclusive in ['neither', 'left']:
            # end_date -= delta
            pass
    else:
        raise ValueError(f"Unsupported start/end/periods combination {start, end, periods}")

    if start_date is not None and end_date is not None:
        if align == 'left':
            dr = _sedl(start_date, end_date, delta)
        elif align == 'right':
            dr = _sedr(start_date, end_date, delta)
        else:
            raise ValueError(f"Unsupported alignment {align}")
    elif start_date is not None and periods is not None:
        dr = _sdp(start_date, periods, delta)
    elif end_date is not None and periods is not None:
        dr = _edp(end_date, periods, delta)
    else:
        raise ValueError(f"Unsupported start/end/periods combination {start, end, periods}")

    if inclusive in ['neither', 'right']:
        dr = dr[1:]
    if inclusive in ['neither', 'left']:
        dr = dr[:-1]

    return pd.DatetimeIndex(data=dr, name=name)
# end
