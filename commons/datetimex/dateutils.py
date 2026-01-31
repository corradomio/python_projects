from typing import Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ---------------------------------------------------------------------------
# is_leap_year
# last_day_of
# to_doy
# to_woy
# to_wom
# ---------------------------------------------------------------------------

#
# La prima settimana dell'anno
# inizia il lunedì della settimana che contiene il primo giovedì dell'anno,
# secondo lo standard ISO 8601.
# Per il 2025, questo significa che la prima settimana è iniziata il lunedì
# 30 dicembre 2024 e terminerà il 5 gennaio 2025
#

#
# La prima settimana del mese e' qualunque settimana che inizia con il
# giorno 1 del mese. La settimana inizia lunedi e finisce domenica.
# Se il primo giorno del mese e' domenica, la settiama e' composa SOLO
# da domenica
#

MONTH_DAYS = [
    [0, 31,28,31,30,31,30,31,31,30,31,30,31,],
    [0, 31,29,31,30,31,30,31,31,30,31,30,31,],
]


def is_leap_year(year):
    return year % 4 == 0 and year % 100 != 0 or year % 400 == 0


def last_day_of(year: int, month: int) -> int:
    return MONTH_DAYS[is_leap_year(year)][month]


def to_doy(dt: datetime) -> int:
    """day of year"""
    year = dt.year
    month = dt.month
    day = dt.day
    leap = is_leap_year(year)
    doy = 0
    mdays = MONTH_DAYS[leap]
    for i in range(month):
        doy += mdays[i]
    doy += day
    return doy
# end


def to_woy(dt: datetime) ->int:
    """week of year"""
    return dt.isocalendar()[1]


def to_wom(dt: datetime) -> int:
    """week of month"""
    dow = datetime(dt.year, dt.month, 1).weekday()
    return (dt.day-dow+6)//7 + 1


# ---------------------------------------------------------------------------
# relativeperiods
# relativedifference
# datetime_difference
# ---------------------------------------------------------------------------



#
#     ‘B’: Business Day
#     ‘D’: Calendar day
#     ‘W’: Weekly
#     ‘M’: Month end
#     ‘BM’: Business month end
#     ‘MS’: Month start
#     ‘BMS’: Business month start
#     ‘Q’: Quarter end
#     ‘BQ’: Business quarter end
#     ‘QS’: Quarter start
#     ‘BQS’: Business quarter start
#     ‘A’ or ‘Y’: Year end
#     ‘BA’ or ‘BY’: Business year end
#     ‘AS’ or ‘YS’: Year start
#     ‘BAS’ or ‘BYS’: Business year start
#     ‘H’: Hourly
#     ‘T’ or ‘min’: Minutely
#     ‘S’: Secondly
#     ‘L’ or ‘ms’: Milliseconds
#     ‘U’: Microseconds
#     ‘N’: Nanoseconds
#
# years, months, weeks, days, hours, minutes, seconds, microseconds:

# year with 365/366 days
#      12*30 = 360

FREQ_TO_PERIOD = {
    'D': 'days',
    'B': 'days',
    'W': 'weeks',
    'M': 'months',
    'BM': 'months',
    'MS': 'months',
    'BMS': 'months',
    'A': 'years',
    'Y': 'years',
    'BA': 'years',
    'AS': 'years',
    'H': 'hours',
    'T': 'minutes',
    'min': 'minutes',
    'S': 'seconds',
    'L': 'milliseconds',
    'ms': 'milliseconds',
    'U': 'microseconds',
    'N': 'nanoseconds'
}


# ---------------------------------------------------------------------------
# relativeperiods
# relativedifference
# datetime_difference
# ---------------------------------------------------------------------------

def relativeperiods(periods=1, freq='D'):
    """
    As relative delta but it is possible to use Pandas periods & frequencies
    :param periods: n of periods
    :param freq: Pandas frequency
    :return: a 'relativedelta' object
    """
    return relativedelta(**{
        FREQ_TO_PERIOD[freq]: periods
    })


def relativedifference(dt1: datetime, dt2: datetime, freq='D') -> int:
    dd: timedelta = dt1 - dt2
    if freq == 'D':
        return dd.days
    if freq == 'W':
        # add 6 to ensure COMPLETE weeks
        if dd.days < 0:
            return (dd.days)//7             # WARNING: (-8)//7 == -2
        else:
            return (dd.days + 6)//7         # WARNING: (+8)//7 ==  1
    if freq == 'M':
        return 12*(dt1.year - dt2.year) + (dt1.month - dt2.month)
    else:
        raise ValueError(f"Unsupported freq {freq}")


def datetime_difference(dt1: datetime, dt2: datetime, freq='D') -> float:
    dd: timedelta = dt1 - dt2
    if freq in ['S', 's', 'sec']:
        return dd.total_seconds()
    if freq in ['m', 'min']:
        return dd.total_seconds() / 60
    if freq in ['h', 'H']:
        return dd.total_seconds() / 3600
    if freq == 'D':
        return dd.total_seconds() / 86400
    if freq == 'W':
        return dd.total_seconds() / (7*86400)
    if freq == 'M':
        if dt1.day == 1 and dt2.day == 1:
            return 12. * (dt1.year - dt2.year) + (dt1.month - dt2.month)
        if dt1.day == last_day_of(dt1.year, dt1.month) and dt2.day == last_day_of(dt2.year, dt2.month):
            return 12. * (dt1.year - dt2.year) + (dt1.month - dt2.month)
        else:
            return 12. * (dt1.year - dt2.year) + (dt1.month - dt2.month) + (dt1.day - dt2.day)/last_day_of(dt1.year, dt1.month)
    else:
        raise ValueError(f"Unsupported freq {freq}")

#
#
#     Offset aliases
#     --------------
#
#         B       business day frequency
#         C       custom business day frequency
#         D       calendar day frequency
#         W       weekly frequency
#         ME      month end frequency
#         SME     semi-month end frequency (15th and end of month)
#         BME     business month end frequency
#         CBME    custom business month end frequency
#         MS      month start frequency
#         SMS     semi-month start frequency (1st and 15th)
#         BMS     business month start frequency
#         CBMS    custom business month start frequency
#         QE      quarter end frequency
#         BQE     business quarter end frequency
#         QS      quarter start frequency
#         BQS     business quarter start frequency
#         YE      year end frequency
#         BYE     business year end frequency
#         YS      year start frequency
#         BYS     business year start frequency
#         h       hourly frequency
#         bh      business hour frequency
#         cbh     custom business hour frequency
#         min     minutely frequency
#         s       secondly frequency
#         ms      milliseconds
#         us      microseconds
#         ns      nanoseconds
#
#     Deprecated      H, BH, CBH, T,   S, L,  U,  N
#     Alternatives    h, bh, cbh, min, s, ms, us, ns
#
#
#     Period aliases
#     --------------
#
#         B   business day frequency
#         D   calendar day frequency
#         W   weekly frequency
#         M   monthly frequency
#         Q   quarterly frequency
#         Y   yearly frequency
#         h   hourly frequency
#         min minutely frequency
#         s   secondly frequency
#         ms  milliseconds
#         us  microseconds
#         ns  nanoseconds
#
#     Deprecated      A, H, T,   S, L,  U,  N
#     Alternatives    Y, h, min, s, ms, us, ns
#
#
#     Combining aliases
#     -----------------
#
#         <day><intraday>
#
#
#     Anchored offsets
#     ----------------
#
#     W-SUN   weekly frequency (Sundays). Same as ‘W’
#     W-MON   weekly frequency (Mondays)
#     W-TUE   weekly frequency (Tuesdays)
#     W-WED   weekly frequency (Wednesdays)
#     W-THU   weekly frequency (Thursdays)
#     W-FRI   weekly frequency (Fridays)
#     W-SAT   weekly frequency (Saturdays)
#
#     (B)Q(E)(S)-DEC  quarterly frequency, year ends in December. Same as ‘QE’
#     (B)Q(E)(S)-JAN  quarterly frequency, year ends in January
#     (B)Q(E)(S)-FEB  quarterly frequency, year ends in February
#     (B)Q(E)(S)-MAR  quarterly frequency, year ends in March
#     (B)Q(E)(S)-APR  quarterly frequency, year ends in April
#     (B)Q(E)(S)-MAY  quarterly frequency, year ends in May
#     (B)Q(E)(S)-JUN  quarterly frequency, year ends in June
#     (B)Q(E)(S)-JUL  quarterly frequency, year ends in July
#     (B)Q(E)(S)-AUG  quarterly frequency, year ends in August
#     (B)Q(E)(S)-SEP  quarterly frequency, year ends in September
#     (B)Q(E)(S)-OCT  quarterly frequency, year ends in October
#     (B)Q(E)(S)-NOV  quarterly frequency, year ends in November
#
#     (B)Y(E)(S)-DEC  annual frequency, anchored end of December. Same as ‘YE’
#     (B)Y(E)(S)-JAN  annual frequency, anchored end of January
#     (B)Y(E)(S)-FEB  annual frequency, anchored end of February
#     (B)Y(E)(S)-MAR  annual frequency, anchored end of March
#     (B)Y(E)(S)-APR  annual frequency, anchored end of April
#     (B)Y(E)(S)-MAY  annual frequency, anchored end of May
#     (B)Y(E)(S)-JUN  annual frequency, anchored end of June
#     (B)Y(E)(S)-JUL  annual frequency, anchored end of July
#     (B)Y(E)(S)-AUG  annual frequency, anchored end of August
#     (B)Y(E)(S)-SEP  annual frequency, anchored end of September
#     (B)Y(E)(S)-OCT  annual frequency, anchored end of October
#     (B)Y(E)(S)-NOV  annual frequency, anchored end of November
#

# Weekday: Monday = 0, Sunday = 6


def _clip_to_dow(dt: datetime, wd: int) -> datetime:
    dt = dt.replace(microsecond=0, second=0, minute=0, hour=0)
    dow = dt.weekday()
    ddays = dow - wd if dow >= wd else 7 + dow - wd
    return dt - relativeperiods(periods=ddays, freq="D")


def _clip_to_month(dt: datetime, month: int, last=False) -> datetime:
    m = dt.month
    if month <= m:
        day = last_day_of(dt.year, month) if last else 1
        return dt.replace(microsecond=0, second=0, minute=0, hour=0, day=day, month=month)
    else:
        day = last_day_of(dt.year - 1, month) if last else 1
        return dt.replace(microsecond=0, second=0, minute=0, hour=0, day=day, month=month, year=dt.year-1)


CLIP_DATETIME = {
    None: lambda dt: dt,

    "N": lambda dt: dt,
    "U": lambda dt: dt,
    "L": lambda dt: dt,
    "ms": lambda dt: dt,

    "S": lambda dt: dt.replace(microsecond=0),
    "s": lambda dt: dt.replace(microsecond=0),
    "sec": lambda dt: dt.replace(microsecond=0),

    "T": lambda dt: dt.replace(microsecond=0, second=0),
    "m": lambda dt: dt.replace(microsecond=0, second=0),
    "min": lambda dt: dt.replace(microsecond=0, second=0),

    "H": lambda dt: dt.replace(microsecond=0, second=0, minute=0),
    "h": lambda dt: dt.replace(microsecond=0, second=0, minute=0),

    "D": lambda dt: dt.replace(microsecond=0, second=0, minute=0, hour=0),
    "DS": lambda dt: dt.replace(microsecond=0, second=0, minute=0, hour=0),
    "DE": lambda dt: dt.replace(microsecond=0, second=59, minute=59, hour=23),

    "W": lambda dt: _clip_to_dow(dt, 6),
    "WS": lambda dt: _clip_to_dow(dt, 0),
    "WE": lambda dt: _clip_to_dow(dt, 6),
    "W-MON": lambda dt: _clip_to_dow(dt, 0),
    "W-TUE": lambda dt: _clip_to_dow(dt, 1),
    "W-WED": lambda dt: _clip_to_dow(dt, 2),
    "W-THU": lambda dt: _clip_to_dow(dt, 3),
    "W-FRI": lambda dt: _clip_to_dow(dt, 4),
    "W-SAT": lambda dt: _clip_to_dow(dt, 5),
    "W-SUN": lambda dt: _clip_to_dow(dt, 6),

    "M": lambda dt: dt.replace(microsecond=0, second=0, minute=0, hour=0, day=1),
    "MS": lambda dt: dt.replace(microsecond=0, second=0, minute=0, hour=0, day=1),
    "ME": lambda dt: dt.replace(microsecond=0, second=0, minute=0, hour=0, day=last_day_of(dt.year, dt.month)),

    "Y": lambda dt: _clip_to_month(dt, 1),
    "Y-JAN": lambda dt: _clip_to_month(dt, 1),
    "Y-FEB": lambda dt: _clip_to_month(dt, 2),
    "Y-MAR": lambda dt: _clip_to_month(dt, 3),
    "Y-APR": lambda dt: _clip_to_month(dt, 4),
    "Y-MAY": lambda dt: _clip_to_month(dt, 5),
    "Y-JUN": lambda dt: _clip_to_month(dt, 6),
    "Y-JUL": lambda dt: _clip_to_month(dt, 7),
    "Y-AUG": lambda dt: _clip_to_month(dt, 8),
    "Y-SEP": lambda dt: _clip_to_month(dt, 9),
    "Y-OCT": lambda dt: _clip_to_month(dt, 10),
    "Y-NOV": lambda dt: _clip_to_month(dt, 11),
    "Y-DEC": lambda dt: _clip_to_month(dt, 12),

    "YS": lambda dt: _clip_to_month(dt, 1),
    "YS-JAN": lambda dt: _clip_to_month(dt, 1),
    "YS-FEB": lambda dt: _clip_to_month(dt, 2),
    "YS-MAR": lambda dt: _clip_to_month(dt, 3),
    "YS-APR": lambda dt: _clip_to_month(dt, 4),
    "YS-MAY": lambda dt: _clip_to_month(dt, 5),
    "YS-JUN": lambda dt: _clip_to_month(dt, 6),
    "YS-JUL": lambda dt: _clip_to_month(dt, 7),
    "YS-AUG": lambda dt: _clip_to_month(dt, 8),
    "YS-SEP": lambda dt: _clip_to_month(dt, 9),
    "YS-OCT": lambda dt: _clip_to_month(dt, 10),
    "YS-NOV": lambda dt: _clip_to_month(dt, 11),
    "YS-DEC": lambda dt: _clip_to_month(dt, 12),

    "YE-JAN": lambda dt: _clip_to_month(dt, 1, True),
    "YE-FEB": lambda dt: _clip_to_month(dt, 2, True),
    "YE-MAR": lambda dt: _clip_to_month(dt, 3, True),
    "YE-APR": lambda dt: _clip_to_month(dt, 4, True),
    "YE-MAY": lambda dt: _clip_to_month(dt, 5, True),
    "YE-JUN": lambda dt: _clip_to_month(dt, 6, True),
    "YE-JUL": lambda dt: _clip_to_month(dt, 7, True),
    "YE-AUG": lambda dt: _clip_to_month(dt, 8, True),
    "YE-SEP": lambda dt: _clip_to_month(dt, 9, True),
    "YE-OCT": lambda dt: _clip_to_month(dt, 10, True),
    "YE-NOV": lambda dt: _clip_to_month(dt, 11, True),
    "YE-DEC": lambda dt: _clip_to_month(dt, 12, True),
    "YE": lambda dt: _clip_to_month(dt, 12, True),
}


def clip_date(dt: datetime, freq: str) -> datetime:
    if freq in CLIP_DATETIME:
        dt = CLIP_DATETIME[freq](dt)
    return dt


def now(freq: Optional[str] = None, tz=None) -> datetime:
    """
    As datetime.now but it is possible to specify the resolution
    using Pandas frequency
    :param freq: Pandas frequency
    :return: a datetime object at the specified resolution
    """
    # year, month=None, day=None, hour=0, minute=0, second=0, microsecond=0
    now_: datetime = datetime.now(tz=tz)
    return clip_date(now_, freq)
    # if freq in ['N', 'U', 'ms', 'L']:
    #     pass
    # elif freq == 'S':
    #     now_ = now_.replace(microsecond=0)
    # elif freq in ['T', 'min']:
    #     now_ = now_.replace(microsecond=0, second=0)
    # elif freq == 'H':
    #     now_ = now_.replace(microsecond=0, second=0, minute=0)
    # elif freq in ['D', 'B', 'W']:
    #     now_ = now_.replace(microsecond=0, second=0, minute=0, hour=0)
    # elif freq in ['M', 'BM', 'MS', 'BMS']:
    #     now_ = now_.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    # elif freq in ['A', 'Y', 'BA', 'AS']:
    #     now_ = now_.replace(microsecond=0, second=0, minute=0, hour=0, day=1, month=1)
    # return now_

