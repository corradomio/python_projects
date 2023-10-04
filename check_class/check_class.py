import pandas as pd
import numpy as np


class C:

    @property
    def week(self):
        return 0


def dtix_week(self: pd.DatetimeIndex):
    return pd.Index(self.isocalendar().week.to_numpy(dtype=np.int32))


def main():

    c = C

    if not hasattr(pd.DatetimeIndex, 'week'):
        pd.DatetimeIndex.week = property(fget=dtix_week)


    print(hasattr(pd.DatetimeIndex, 'week'))

    print(pd.__version__)
    p = pd.period_range("2023-10-01", periods=60, freq='D')
    ix = pd.PeriodIndex(p)

    p = pd.date_range("2023-10-01", periods=60, freq='D')
    ix = pd.DatetimeIndex(p)

    ixs = ix.second

    # print(ix.second)
    # print(ix.minute)
    # print(ix.hour)
    #
    # print(ix.day)
    print(ix.week)
    # print(ix.month)
    # print(ix.quarter)
    print(ix.year)
    pass


if __name__ == "__main__":
    main()