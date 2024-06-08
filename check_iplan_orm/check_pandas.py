import typing
from stdlib import is_instance
from stdlib import dict
import pandas as pd
import pandasx as pdx
import pandas._libs.tslibs.period


class C:
    pass


def main():
    #
    # def c_print(self, what):
    #     print(self, what)
    #
    # C.print = c_print
    #
    # c = C()
    # c.print("ciccio")
    #

    pr = pd.period_range('2024-01-01', periods=12, freq='M')
    dr = pd.date_range('2024-01-01', periods=12, freq='MS')

    prs: pd.Series[pd.Period] = pr.to_series(name='period').reset_index(drop=True)
    drs: pd.Series[pd.Timestamp] = dr.to_series(name='date').reset_index(drop=True)

    d = dict({
        1: 'a'
    })

    df = pd.DataFrame({prs.name: prs, drs.name:drs})

    # assert is_instance(d, dict[int, str])

    p1: pandas._libs.tslibs.period.Period
    p2: pandas.Period

    assert is_instance(prs, pd.Series[pd.Period]), "Invalid pd.Series[pd.Period]"
    assert is_instance(drs, pd.Series[pd.Timestamp]), "Invalid pd.Series[pd.Timestamp]"
    assert is_instance(df, pd.DataFrame[pd.Period, pd.Timestamp]), "pd.DataFrame[pd.Period, pd.Timestamp]"

    pass


if __name__ == "__main__":
    main()
