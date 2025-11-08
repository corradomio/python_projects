from typing import Literal, Union, cast
from stdlib.is_instance import is_instance
from datetime import datetime
from .convert import DT_TYPES, convert

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

DOW_NAME = [[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday"
], [
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun"
]]

MONTH_NAME = [[
    "---",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
], [
    "---",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
]]


DT_FIELD = Literal[
    "year", "month", "day", "hour", "minute", "second", "timestamp",
    "timezone", "tzinfo",
    "day_of_week", "dow", "weekday",
    "day_of_year", "doy",
    "week_of_year", "woy",
    "week_of_month", "wom"
]


MONTH_DAYS = [
    [0, 31,28,31,30,31,30,31,31,30,31,30,31,],
    [0, 31,29,31,30,31,30,31,31,30,31,30,31,],
]

def is_leap(year: int) -> bool:
    """is leap year"""
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    else:
        return year % 4 == 0
# end


def to_doy(dt: datetime) -> int:
    """day of year"""
    year = dt.year
    month = dt.month
    day = dt.day
    leap = is_leap(year)
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



DT_EXTRACTORS = {
    "year": lambda dt: cast(datetime, dt).year,
    "month": lambda dt: cast(datetime, dt).month,
    "day": lambda dt: cast(datetime, dt).day,
    "hour": lambda dt: cast(datetime, dt).hour,
    "minute": lambda dt: cast(datetime, dt).minute,
    "second": lambda dt: cast(datetime, dt).second,
    "timestamp": lambda dt: cast(datetime, dt).timestamp(),

    "timezone": lambda dt: cast(datetime, dt).tzinfo,
    "tzinfo": lambda dt: cast(datetime, dt).tzinfo,

    "day_of_week": lambda dt: cast(datetime, dt).weekday(),
    "dow": lambda dt: cast(datetime, dt).weekday(),
    "weekday": lambda dt: cast(datetime, dt).weekday(),

    "day_of_year": lambda dt: to_doy(dt),
    "doy":  lambda dt: to_doy(dt),

    "week_of_year": lambda dt: to_woy(dt),
    "woy": lambda dt: to_woy(dt),

    "week_of_month": lambda dt: to_wom(dt),
    "wom": lambda dt: to_wom(dt),
}


def extract(dt: DT_TYPES, field: DT_FIELD) -> Union[int, bool, list]:
    if isinstance(field, (list, tuple)):
        fields = field
        return [
            extract(dt, field) for field in fields
        ]
    assert is_instance(dt, DT_TYPES)
    assert is_instance(field, DT_FIELD)
    dt = convert(dt, datetime)
    return DT_EXTRACTORS[field](dt)
