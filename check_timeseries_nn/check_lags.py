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
        # ignore=['DATE'],

        sep=';'
    )
    pe = ppx.PeriodicEncoder(TARGET, datetime='DATE',
                             periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
                             periods="onehot",
                             means=True)
    dfx = pe.fit_transform(df)

    lt = ppx.LagsTransformer(target='EASY', xlags=[0], ylags=[0])
    X, y = lt.fit_transform(dfx)
    pass


if __name__ == "__main__":
    main()
