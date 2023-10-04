import pandas as pd
import pandasx as pdx
import pandasx.preprocessing as ppx

TARGET = 'EASY'
DATETIME = 'DATE'


def main():
    df: pd.DataFrame = pdx.read_data(
        "easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=DATETIME,
        ignore_unnamed=True,
        ignore=DATETIME,

        sep=';'
    )

    X, y = pdx.xy_split(df, target=TARGET)

    X_train, X__test, y_train, y__test = pdx.train_test_split(X, y, test_size=24)

    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
                             # datetime='DATE',
                             # method="sincos",
                             means=True)

    Xp_train = pe.fit_transform(X_train, y_train)
    Xp__test = pe.transform(X__test)

    lt = ppx.LagsTransformer(xlags=[0, 1], ylags=16, tlags=[0, 2, 4])
    Xl_train, yl_train = lt.fit_transform(Xp_train, y_train)
    Xl__test, yl__test = lt.transform(Xp__test, y__test)

    at = ppx.ArrayTransformer(xlags=[0, 1, 2], ylags=[0, 1, 2], tlags=[0, 1], sequence=False, swap=False)
    Xt_train0, yt_train0 = at.fit_transform(Xl_train, yl_train)
    Xt__test0, yt__test0 = at.transform(Xl__test, yl__test)

    at = ppx.ArrayTransformer(xlags=[0, 1, 2], ylags=[0, 1, 2], tlags=[0, 1], sequence=True, swap=False)
    Xt_train1, yt_train1 = at.fit_transform(Xl_train, yl_train)
    Xt__test1, yt__test1 = at.transform(Xl__test, yl__test)

    at = ppx.ArrayTransformer(xlags=[0, 1, 2], ylags=[0, 1, 2], tlags=[0, 1], sequence=True, swap=True)
    Xt_train2, yt_train2 = at.fit_transform(Xl_train, yl_train)
    Xt__test2, yt__test2 = at.transform(Xl__test, yl__test)

    pass


if __name__ == "__main__":
    main()
