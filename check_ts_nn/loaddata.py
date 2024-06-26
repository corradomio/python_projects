import numpy as np
import pandas as pd

import pandasx as pdx
import pandasx.preprocessing as ppx


def lags_transform(X, y):
    at = ppx.LagsArrayTransformer(xlags=[0, 1], ylags=[], tlags=[0, 1],
                                  temporal=False,
                                  channels=False,
                                  y_flatten=True,
                                  dtype=np.float32)
    Xt, yt, it = at.fit(X, y).transform(X, y)
    pass


def load_data(tlags=24):
    df: pd.DataFrame = pdx.read_data(
        "easy_ts.csv",
        datetime=('DATE', '%Y-%m-%d', 'M'),
        index=['DATE'],
        ignore_unnamed=True,
        ignore=['DATE'],

        sep=';'
    )

    train, test_ = pdx.train_test_split(df, test_size=24)

    pe = ppx.PeriodicEncoder(periodic=ppx.PERIODIC_MONTH | ppx.PERIODIC_QUARTER,
                             datetime=None,
                             target='EASY',
                             method=None,
                             means=True)

    train_p = pe.fit_transform(train)
    test__p = pe.transform(test_)

    lt = ppx.LagsTransformer(xlags=[0], ylags=range(17), target='EASY')
    train_l = lt.fit_transform(train_p)
    test__l = lt.transform(test__p)

    X_train, y_train, X_test_, y_test_ = pdx.xy_split(train_l, test__l, target='EASY', shared='EASY')

    scx = ppx.StandardScaler()
    Xs_train = scx.fit_transform(X_train)
    Xs_test_ = scx.transform(X_test_)

    scy = ppx.StandardScaler()
    ys_train = scy.fit_transform(y_train)
    ys_test_ = scy.transform(y_test_)
    # Xs_train, ys_train, Xs_test_, ys_test_ = X_train, y_train, X_test_, y_test_

    # lags_transform(Xs_train, ys_train)

    at = ppx.LagsArrayTransformer(xlags=24, ylags=0, tlags=tlags,
                                  temporal=True,
                                  y_flatten=False,
                                  dtype=np.float32)
    Xt, yt, it = at.fit(Xs_train, ys_train).transform(Xs_train, ys_train)
    # Xp, yp, ip = at.transform(Xs_test_, ys_test_)

    return Xt, yt, it, ys_train, Xs_test_, ys_test_, at
