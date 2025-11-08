from datetime import date, datetime
from numpy import datetime64
from pandas import Timestamp
from datetimex import convert, extract, DT_EXTRACTORS

def main():
    this_string = '2024-07-13 03:03:01'
    dt = convert(this_string, datetime)

    print(dt.timetuple())

    for field in DT_EXTRACTORS.keys():
        print(field, ":", extract(dt, field))

    pass


if __name__ == "__main__":
    main()
