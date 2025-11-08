#
# All possible conversions between
#
#       str, int, date, datetime, timestamp, datetime64, Timestamp, Period
#
__all__ = [
    "convert",
    "to_datetime",
    "to_period",
    "DT_CONVERTERS"
]

from deprecated import deprecated
from typing import Union, Callable
from datetime import date, time, datetime, timedelta, timezone
from numpy import datetime64, timedelta64
from pandas import Timestamp, Timedelta, Period
from stdlib.is_instance import is_instance


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

DT_TYPES = Union[str, int, date, time, datetime, datetime64, Timestamp]
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'
UNIX_DATETIME64 = datetime64('1970-01-01T00:00:00')
UNIX_TIMEDELTA = timedelta64(1, 's')
TZINFO_LOCAL = datetime.now().astimezone().tzinfo
TZINFO_DELTA = TZINFO_LOCAL.utcoffset(None)
UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
UNIX_DATE = UNIX_EPOCH.date()


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
#
# If not specified, 'date' and 'datetime' use the CURRENT timezone
# datetime64 and Timestamp use 'GMT+0'
#

#
#   class Timestamp(datetime)
#       Timestamp extends datetime
#
#
#   Note: IF tsinfo is not specified, it is used  the
#         CURRENT tzinfo
#
#   2024/07/03 03:03:01
#       -> 1720825381
#           Sat Jul 13 2024 03:03:01 GMT+0400 (Gulf Standard Time)
#           Fri Jul 12 2024 23:03:01 GMT+0000


def guess_format(string):
    # 2024/01/01 13:30:45
    # 2024-01-01 13:30:45
    has_slash = string.find('/') > 0
    has_colon = string.find(':') > 0
    has_ti = string.find('T') > 0
    if not has_colon:
        return '%Y/%m/%d' if has_slash else '%Y-%m-%d'
    if has_slash:
        return '%Y/%m/%dT%H:%M:%S' if has_ti else '%Y/%m/%d %H:%M:%S'
    else:
        return '%Y-%m-%dT%H:%M:%S' if has_ti else '%Y-%m-%d %H:%M:%S'


# timestamp: number of seconds from '1970-01-01 00:00:00'

# -- date -> [datetime, str, timestamp, datetime64]

def date_to_datetime(from_date: date) -> datetime:
    return datetime(from_date.year, from_date.month, from_date.day)


def date_to_str(from_date: date, to_format=None) -> str:
    if to_format is None:
        to_format = DATE_FORMAT
    return from_date.strftime(to_format)


def date_to_timestamp(from_date: date) -> int:
    from_datetime = datetime(from_date.year, from_date.month, from_date.day, tzinfo=TZINFO_LOCAL)
    return int((from_datetime - UNIX_EPOCH).total_seconds())

# -- datetime -> [date, str, timestamp, datetime64]

def datetime_to_date(from_datetime: datetime) -> date:
    return from_datetime.date()


# def datetime_to_datetime(from_datetime: datetime, to_format=None) -> datetime:
#     if to_format is None:
#         to_format = DATETIME_FORMAT
#     return datetime.strptime(from_datetime.strftime(to_format), to_format)


def datetime_to_str(from_datetime: datetime, to_format=None) -> str:
    if to_format is None:
        to_format = DATETIME_FORMAT
    return from_datetime.strftime(to_format)


def datetime_to_timestamp(from_datetime: datetime) -> int:
    return int(from_datetime.timestamp())


# -- string -> [date, datetime, timestamp, datetime64]

def str_to_date(string: str, current_format=None) -> date:
    if current_format is None:
        current_format = guess_format(string)
    return datetime.strptime(string, current_format).date()


def str_to_datetime(string, current_format=None) -> datetime:
    if current_format is None:
        current_format = guess_format(string)
    return datetime.strptime(string, current_format)


# def str_to_str(string, current_format=None, to_format=None):
#     if current_format is None and to_format is not None:
#         current_format = DATE_FORMAT if len(string) <= 10 else DATETIME_FORMAT
#     if current_format is not None and to_format is not None:
#         return datetime.strptime(string, current_format).strftime(to_format)
#     else:
#         return string


def str_to_timestamp(string, current_format=None):
    if current_format is None:
        current_format = guess_format(string)
    return int(datetime.strptime(string, current_format).timestamp())


# -- timestamp (as integer) -> [date, datetime, str, datetime64]

def timestamp_to_date(timestamp):
    return datetime.fromtimestamp(timestamp).date()


def timestamp_to_datetime(timestamp):
    return datetime.fromtimestamp(timestamp)


def timestamp_to_str(timestamp, to_format=None):
    return datetime_to_str(timestamp_to_datetime(timestamp), to_format)


# -- datetime64 -> [str, date, datetime, timestamp] -> datetime64
#
#   from string or Unix epoch (offset from 1 January 1970)

def datetime64_to_str(dt64: datetime64, to_format=None):
    if to_format is None:
        to_format = DATETIME_FORMAT
    return datetime64_to_datetime(dt64).strftime(to_format)


def datetime64_to_date(dt64: datetime64):
    timestamp: int = int((dt64 - UNIX_DATETIME64) / UNIX_TIMEDELTA)
    return date.fromtimestamp(timestamp)


def datetime64_to_datetime(dt64: datetime64):
    tzoffset = TZINFO_DELTA.seconds
    timestamp: int = int((dt64 - UNIX_DATETIME64) / UNIX_TIMEDELTA) - tzoffset
    return datetime.fromtimestamp(timestamp)


def datetime64_to_timestamp(dt64: datetime64):
    return int((dt64 - UNIX_DATETIME64) / UNIX_TIMEDELTA)


def date_to_datetime64(from_date: date):
    tzoffset = TZINFO_DELTA.seconds
    return datetime64(date_to_timestamp(from_date) + tzoffset, 's')


def datetime_to_datetime64(from_datetime: datetime):
    return datetime64(datetime_to_timestamp(from_datetime), 's')


def str_to_datetime64(string: str, current_format=None):
    return datetime_to_datetime64(str_to_datetime(string, current_format))


def timestamp_to_datetime64(timestamp: int):
    tzoffset = TZINFO_DELTA.seconds
    return datetime64(timestamp+tzoffset, 's')



# -- Timestamp ->  [str, date, datetime, datetime64, timestamp] -> Timestamp
#
#   from string or Unix epoch (offset from 1 January 1970)

def Timestamp_to_str(ts: Timestamp, to_format=None):
    return datetime_to_str(ts.to_pydatetime(), to_format)


def Timestamp_to_date(ts: Timestamp):
    return datetime_to_date(ts.to_pydatetime())


def Timestamp_to_datetime(ts: Timestamp):
    return ts.to_pydatetime()


def Timestamp_to_datetime64(ts: Timestamp):
    return ts.to_datetime64()


def Timestamp_to_timestamp(ts: Timestamp):
    return int(ts.timestamp())

# --

def str_to_Timestamp(string: str, current_format=None):
    dt64 = str_to_datetime64(string, current_format)
    return Timestamp(dt64)


def date_to_Timestamp(from_date: date):
    dt64 = date_to_datetime64(from_date)
    return Timestamp(dt64)


def datetime_to_Timestamp(from_datetime: datetime):
    dt64 = datetime_to_datetime64(from_datetime)
    return Timestamp(dt64)


def datetime64_to_Timestamp(dt64: datetime64):
    return Timestamp(dt64)


def timestamp_to_Timestamp(timestamp: int):
    return Timestamp(timestamp_to_datetime64(timestamp))


# -- aliases

int_to_date = timestamp_to_date
int_to_datetime = timestamp_to_datetime
int_to_datetime64 = timestamp_to_datetime64
int_to_str = timestamp_to_str
int_to_Timestamp = timestamp_to_Timestamp

# -- identity

def _dt_identity(dt, **kwargs):
    return dt

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

DT_CONVERTERS: dict[tuple[type, type], Callable] = {
    (str, str): _dt_identity,
    (str, date): str_to_date,
    (str, datetime): str_to_datetime,
    (str, int): str_to_timestamp,
    (str, datetime64): str_to_datetime64,
    (str, Timestamp): str_to_Timestamp,

    (date, str): date_to_str,
    (date, date): _dt_identity,
    (date, datetime): date_to_datetime,
    (date, int): date_to_timestamp,
    (date, datetime64): date_to_datetime64,
    (date, Timestamp): date_to_Timestamp,

    (datetime, str): datetime_to_str,
    (datetime, date): datetime_to_date,
    (datetime, datetime): _dt_identity,
    (datetime, int): datetime_to_timestamp,
    (datetime, datetime64): datetime_to_datetime64,
    (datetime, Timestamp): datetime_to_Timestamp,

    (datetime64, str): datetime64_to_str,
    (datetime64, date): datetime64_to_date,
    (datetime64, datetime): datetime64_to_datetime,
    (datetime64, int): datetime64_to_timestamp,
    (datetime64, datetime64): _dt_identity,
    (datetime64, Timestamp): datetime64_to_Timestamp,

    (Timestamp, str): Timestamp_to_str,
    (Timestamp, date): Timestamp_to_date,
    (Timestamp, datetime): Timestamp_to_datetime,
    (Timestamp, int): Timestamp_to_timestamp,
    (Timestamp, datetime64): Timestamp_to_datetime64,
    (Timestamp, Timestamp): _dt_identity,

    (int, str): timestamp_to_str,
    (int, date): timestamp_to_date,
    (int, datetime): timestamp_to_datetime,
    (int, int): _dt_identity,
    (int, datetime64): timestamp_to_datetime64,
    (int, Timestamp): timestamp_to_Timestamp,
}


def convert(dt: DT_TYPES, to_type: type, **kwargs) -> DT_TYPES:
    assert is_instance(dt, DT_TYPES)
    assert isinstance(to_type, type)

    if isinstance(dt, Period): dt = dt.to_timestamp()
    from_type = type(dt)

    if (from_type, to_type) not in DT_CONVERTERS:
        raise ValueError(f"Unable to convert {from_type} into {to_type}")

    dtcvt = DT_CONVERTERS[(from_type, to_type)]
    return dtcvt(dt, **kwargs)


@deprecated
def to_datetime(dt: DT_TYPES, to_type: type, **kwargs) -> DT_TYPES:
    return convert(dt, to_type, **kwargs)


def to_period(dt: DT_TYPES, freq) -> Period:
    # Period(value)
    #   value: str, date, datetime, Timestamp
    if not isinstance(dt, (str, date, datetime, Timestamp, Period)):
        dt = to_datetime(dt, Timestamp)
    return Period(dt, freq=freq)
