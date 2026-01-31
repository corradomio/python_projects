import datetime as dt
import numpy as np
import datetimex as dtx


def check_convert():

    now = dt.datetime.now()
    today = dt.date.today()

    print(now)          # yyyy-mm-dd HH:MM:SS.uuuuuu
    print(today)        # yyyy-mm-dd

    now64 = dtx.convert(now, np.datetime64)
    today64 = dtx.convert(today, np.datetime64)

    print(now64)        # yyyy-mm-ddTHH:MM:SS.uuuuuu
    print(today64)      # yyyy-mm-dd

    nowpy = dtx.convert(now64, dt.datetime)
    todaypy = dtx.convert(today64, dt.date)

    print(nowpy)
    print(todaypy)

    pass


def check_extract():
    now = dt.datetime.now()

    for field in dtx.DT_EXTRACTORS.keys():
        print(field, ":", dtx.extract(now, field))


def main():
    # check_convert()
    check_extract()


if __name__ == "__main__":
    main()
