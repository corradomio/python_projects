import numpy as np
import pandas as pd
import pandasx as pdx
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from stdlib import qualified_name

from sktimex import LinearForecastRegressor
from sktimex.scikit_model import ScikitForecastRegressor


def main1():
    df = pdx.read_data('../data/stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'),
                       index=['agency', 'sku', 'date'],
                       ignore=['agency', 'sku', 'date', 'timeseries'])

    X, y = pdx.xy_split(df, target='volume')

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, train_size=0.8)
    fh = ForecastingHorizon(np.arange(1, 11))

    # --

    skr = ScikitForecastRegressor(
        class_name=qualified_name(LinearRegression),
        window_length=1,
        strategy='recursive'
    )
    skr.fit(X=X_train, y=y_train)

    y_pred_1 = skr.predict(fh=fh, X=X_test)
    yp_1 = y_pred_1.loc[('Agency_22', 'SKU_01')]

    # --

    lfr = LinearForecastRegressor(
        class_name=qualified_name(LinearRegression),
        lag=dict(
            length=1,
            current=False
        )
    )
    lfr.fit(X=X_train, y=y_train)

    y_pred_2 = lfr.predict(X=X_test, fh=fh)
    yp_2 = y_pred_2.loc[('Agency_22', 'SKU_01')]

    pass


def main():
    df_all = pdx.read_data('../data/stallion_all.csv',
                           datetime=('date', '%Y-%m-%d', 'M'),
                           index=[ 'agency', 'sku', 'date'],
                           ignore=['agency', 'sku', 'date', 'timeseries'])

    df: pd.DataFrame = df_all.loc[('Agency_22', 'SKU_01')]
    df.to_csv("../data.csv")
    df_train, df_test = pdx.train_test_split(df, train_size=0.8)
    X_train, y_train, X_test, y_test = pdx.xy_split(df_train, df_test, target='volume')

    X, y = pdx.xy_split(df, target='volume')

    # --

    fh1 = np.arange(1, len(X_test) + 1)
    fh2 = ForecastingHorizon(X_test.index, is_relative=False)
    fh3 = [3, 5, 7, 9]
    fh4 = None
    fh5 = 9

    # --

    skr = ScikitForecastRegressor(
        class_name=qualified_name(LinearRegression),
        window_length=3,
        strategy='recursive'
    )

    skr.fit(y=y_train, X=X_train)

    y_pred_1 = skr.predict(fh=fh1, X=X_test)
    y_pred_2 = skr.predict(fh=fh2, X=X_test)
    y_pred_3 = skr.predict(fh=fh3, X=X_test)
    y_pred_4 = skr.predict(fh=fh4, X=X_test)
    y_pred_5 = skr.predict(fh=fh5, X=X_test)

    # --

    skr.fit(y=y_train)

    y_pred_6 = skr.predict(fh=fh1)
    y_pred_7 = skr.predict(fh=fh2)
    y_pred_8 = skr.predict(fh=fh3)
    y_pred_9 = skr.predict(fh=fh5)

    # --

    lfr = LinearForecastRegressor(
        class_name=qualified_name(LinearRegression),
        lag=dict(
            length=3,
            current=False
        )
    )

    lfr.fit(y=y_train, X=X_train)

    y_pred_20 = lfr.predict(y=y_train, X=X)


    y_pred_11 = lfr.predict(fh=fh1, X=X_test)
    y_pred_12 = lfr.predict(fh=fh2, X=X_test)
    y_pred_13 = lfr.predict(fh=fh3, X=X_test)
    y_pred_14 = lfr.predict(fh=fh4, X=X_test)
    y_pred_15 = lfr.predict(fh=fh5, X=X_test)

    # --

    lfr.fit(y=y_train)

    y_pred_16 = lfr.predict(fh=fh1)
    y_pred_17 = lfr.predict(fh=fh2)
    y_pred_18 = lfr.predict(fh=fh3)
    y_pred_19 = lfr.predict(fh=fh5)

    pass


if __name__ == "__main__":
    main()
