from typing import Literal, Union, cast
from stdlib.is_instance import is_instance
from datetime import datetime
from .convert import DT_TYPES, convert
from .dateutils import to_doy, to_wom, to_woy


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
