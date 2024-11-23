from typing import Optional
from datetime import datetime

from dateutil.relativedelta import relativedelta

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


def now(freq: Optional[str] = None, tz=None) -> datetime:
    """
    As datetime.now but it is possible to specify the resolution
    using Pandas frequency
    :param freq: Pandas frequency
    :return: a datetime object at the specified resolution
    """
    # year, month=None, day=None, hour=0, minute=0, second=0, microsecond=0
    now_: datetime = datetime.now(tz=tz)
    if freq in ['N', 'U', 'ms', 'L']:
        pass
    elif freq == 'S':
        now_ = now_.replace(microsecond=0)
    elif freq in ['T', 'min']:
        now_ = now_.replace(microsecond=0, second=0)
    elif freq == 'H':
        now_ = now_.replace(microsecond=0, second=0, minute=0)
    elif freq in ['D', 'B', 'W']:
        now_ = now_.replace(microsecond=0, second=0, minute=0, hour=0)
    elif freq in ['M', 'BM', 'MS', 'BMS']:
        now_ = now_.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    elif freq in ['A', 'Y', 'BA', 'AS']:
        now_ = now_.replace(microsecond=0, second=0, minute=0, hour=0, day=1, month=1)
    return now_


def clip_date(dt: datetime, freq: str) -> datetime:
    if freq == 'D':
        dt = dt.replace(microsecond=0, second=0, minute=0, hour=0)
    elif freq == 'M':
        dt = dt.replace(microsecond=0, second=0, minute=0, hour=0, day=1)
    elif freq == 'Y':
        dt = dt.replace(microsecond=0, second=0, minute=0, hour=0, day=1, month=1)
    return dt
