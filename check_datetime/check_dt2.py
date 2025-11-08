from datetime import date, datetime
from numpy import datetime64
from pandas import Timestamp
from datetimex import convert


def main():
    this_string = '2024-07-13 03:03:01'
    this_date = date(2024, 7, 13)
    this_datetime = datetime(2024, 7, 13, 3, 3, 1)
    this_datetime64 = datetime64(int((this_datetime - datetime(1970, 1, 1)).total_seconds()), 's')
    this_Timestamp = Timestamp(this_datetime64)

    dtlist = [
        this_string,
        this_date,
        this_datetime,
        this_datetime64,
        this_Timestamp,
        # 1720825381,
        # 1720814400,
    ]

    for dt in dtlist:
        timestamp = convert(dt, int)
        tt = convert(timestamp, type(dt))
        print(f"{type(dt).__name__}_to_timestamp: {dt} -> {tt}")
        pass

    pass


if __name__ == "__main__":
    main()
