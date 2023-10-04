import pandas as pd
import pandasx as pdx
import pandasx.preprocessing as ppx

TARGET = 'EASY'


# def check_df(df):
#
#     pe = ppx.PeriodicEncoder(TARGET,
#                              datetime='DATE',
#                              periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
#                              method="sincos",
#                              means=True)
#     X = pe.fit(df).transform(df)
#
#     lt = ppx.LagsTransformer(target='EASY', xlags=[0], ylags=[0])
#     X, y = lt.fit(df).transform(df)


def check_Xy(df):

    X, y = pdx.xy_split(df, target=['EASY'])

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=24)

    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
                             # datetime='DATE',
                             # method="sincos",
                             means=True)

    Xp_train = pe.fit_transform(X_train, y_train)

    lt = ppx.LagsTransformer(xlags=[0, 1], ylags=16, tlags=[0, 2, 4])
    Xt, yt = lt.fit_transform(Xp_train, y_train)

    lt = ppx.LagsTransformer(xlags=[0], ylags=[], tlags=[0])
    Xt, yt = lt.fit_transform(Xp_train, y_train)

    at = ppx.ArrayTransformer(xlags=[0], ylags=[], tlags=[0], swap=True)
    Xtt, ytt = at.fit_transform(Xt, yt)

    at = ppx.ArrayTransformer(xlags=[0,1,2], ylags=[0,1,2], tlags=[0,1,2], flatten=True)
    Xtt, ytt = at.fit_transform(Xt, yt)

    return Xt, yt


def main():

    df: pd.DataFrame = pdx.read_data(
        "easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=['DATE'],
        ignore_unnamed=True,
        ignore=['DATE'],

        sep=';'
    )

    # check_df(df)
    check_Xy(df)

    pass


if __name__ == "__main__":
    main()
