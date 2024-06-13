import pandas as pd
from datetime import datetime, date
from stdlib.dateutilx import relativeperiods


def to_datetime(dt) -> datetime:
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    if isinstance(dt, pd.Timestamp):
        return dt.to_pydatetime()
    if isinstance(dt, date):
        return datetime(dt.year, dt.month, dt.day)
    if isinstance(dt, datetime):
        return dt
    if isinstance(dt, pd.Period):
        return dt.to_timestamp().to_pydatetime()
    else:
        raise ValueError(f"Unsupported datetime {dt}")


def date_range(start=None, end=None, periods=None, freq=None, name=None, inclusive='both') -> pd.DatetimeIndex:
    assert start is not None and freq is not None
    assert end is not None or periods is not None
    assert inclusive in ['left', 'right', 'both', 'neither']

    start = to_datetime(start)
    end = to_datetime(end)

    dr = []
    curr_date = start
    if periods is not None:
        delta = relativeperiods(periods=periods, freq=freq)

        if inclusive in ['left', 'right', ]:
            periods += 1
        elif inclusive == 'neither':
            periods += 2
        for _ in range(periods):
            dr.append(curr_date)
            curr_date += delta
    else:
        delta = relativeperiods(periods=1, freq=freq)

        while curr_date <= end:
            dr.append(curr_date)
            curr_date += delta
        if curr_date != end:
            dr.append(curr_date)
    # end

    if inclusive in ['right', 'neither'] and len(dr) > 0:
        dr = dr[1:]
    if inclusive in ['left', 'neither'] and len(dr) > 0:
        dr = dr[:-1]

    return pd.DatetimeIndex(data=dr)