import pandas as pd
import pandasx as pdx


def main():

    df = pd.DataFrame()

    # dr = pd.date_range(start='2018-01-01', end='2018-12-01', freq='MS')
    # pr = pd.period_range(start='2018-01-01', end='2018-12-01', freq='M')

    dt = pdx.to_datetime(['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01', '2018-07-01', '2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01'], format='%Y-%m-%d')
    dp = dt.to_period('M')

    d2 = dt.to_period()

    p = pdx.to_period('2018-01-01', freq='M')
    pi = pdx.to_period(['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01', '2018-07-01', '2018-08-01', '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01', ], format='%Y-%m-%d')

    # df["dr"] = dr
    # df["pr"] = pr
    df['dt'] = dt
    df['dp'] = dp
    df['pi'] = pi

    pass



if __name__ == "__main__":
    main()
