from datetime import date, datetime
from numpy import datetime64
from pandas import Timestamp
from datetimex import to_datetime
from dateutil.relativedelta import relativedelta


def main():
    dt1 = datetime(2022, 8, 17)
    dt2 = datetime(2025, 8, 11)

    dd = dt1 - dt2

    rd = relativedelta(dt1, dt2)
    print(rd.years)
    print(rd.months)
    print(rd.weeks)
    print(rd.days)

    d3 = dt2 + rd

    d4 = dt2 + relativedelta(years=rd.years, months=rd.months, weeks=rd.weeks)

    pass


if __name__ == "__main__":
    main()
