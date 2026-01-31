from datetime import datetime
from datetimex import relativedifference, relativeperiods, clip_date, datetime_difference


def main():
    # print(datetime_difference(datetime(2024, 7, 13), datetime(2024, 7, 13)))
    # print(datetime_difference(datetime(2024, 7, 13), datetime(2024, 7, 12,23,59,59), freq='D'))
    # print(datetime_difference(datetime(2024, 7, 13), datetime(2024, 7, 12), freq='D'))
    # print(datetime_difference(datetime(2024, 7, 13), datetime(2024, 7, 6), freq='W'))
    # print(datetime_difference(datetime(2024, 7, 13), datetime(2024, 6, 13), freq='M'))

    print(datetime_difference(datetime(2024, 2, 1), datetime(2024, 1, 1), freq='M'))
    print(datetime_difference(datetime(2024, 2, 29), datetime(2024, 1, 31), freq='M'))
    print(datetime_difference(datetime(2024, 2, 10), datetime(2024, 1, 10), freq='M'))
    print(datetime_difference(datetime(2024, 2, 10), datetime(2024, 1, 9), freq='M'))
    print(datetime_difference(datetime(2024, 2, 10), datetime(2024, 1, 11), freq='M'))



if __name__ == "__main__":
    main()
