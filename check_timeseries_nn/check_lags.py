import pandas as pd
import pandasx as pdx
import pandasx.preprocessing as ppx


def main():
    TARGET = 'EASY'

    df: pd.DataFrame = pdx.read_data(
        "easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=['DATE'],
        ignore_unnamed=True,
        ignore=['DATE'],

        sep=';'
    )
    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
                             target=TARGET)
    dfx = pe.fit_transform(df)

    lt = ppx.LagsTransformer(target='EASY', lags=[1, 1])

    dfl = lt.fit_transform(dfx)
    pass


if __name__ == "__main__":
    main()
