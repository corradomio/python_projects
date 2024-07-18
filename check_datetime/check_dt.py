from datetime import date, datetime
from numpy import datetime64
from pandas import Timestamp
from datetimex import to_datetime


def main():
    this_string = '2024-07-13 03:03:01'
    this_date = date(2024, 7, 13)
    this_datetime = datetime(2024, 7, 13, 3, 3, 1)
    this_datetime64 = datetime64('2024-07-13 03:03:01')
    this_timestamp = Timestamp('2024-07-13 03:03:01')

    dtlist = [
        this_string,
        this_date,
        this_datetime,
        this_datetime64,
        this_timestamp,
        1720825381,
        1720814400,
    ]
    dttypes = [
        str, int, date, datetime, datetime64, Timestamp
    ]

    print(to_datetime(this_string, to_type=str, to_format='%Y-%m-%d'))

    for dt in dtlist:
        from_type = type(dt)
        for to_type in dttypes:
            print(f"{from_type.__name__}_to_{to_type.__name__}:")
            print(f"... {dt} ->  {to_datetime(dt, to_type)}")
            pass

    pass


if __name__ == "__main__":
    main()
