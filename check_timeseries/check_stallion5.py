import numpy as np
import pandas as pd
import pandasx as pdx
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from stdlib import qualified_name

from etime import LinearForecastRegressor
from etime.scikit_model import ScikitForecastRegressor


def main():
    df_all = pdx.read_data('./stallion_all.csv',
                       datetime=('date', '%Y-%m-%d', 'M'),
                       index=['agency', 'sku', 'date'],
                       ignore=['agency', 'sku', 'date', 'timeseries'])

    df: pd.DataFrame = df_all.loc[('Agency_22', 'SKU_01')]
    df.to_csv("data.csv")

    df_train, df_test = pdx.train_test_split(df, train_size=0.8)
    X_train, y_train, X_test, y_test = pdx.xy_split(df_train, df_test, target='volume')

    X, y = pdx.xy_split(df, target='volume')

    # --

    skr = ScikitForecastRegressor(
        class_name=qualified_name(LinearRegression),
        window_length=5
    )

    skr.fit(y=y_train, X=X_train)

    y_pred_1 = skr.predict(fh=ForecastingHorizon(y_test.index, is_relative=False), X=X_test)
    y_pred_2 = skr.predict(X=X, y=y_train)
    # y_pred_3 = skr.predict(fh=ForecastingHorizon(y.index, is_relative=False), X=X)
    y_pred_3 = skr.predict(X=X)

    # --

    lfr = LinearForecastRegressor(
        class_name=qualified_name(LinearRegression),
        lag=dict(
            length=5,
            current=False
        )
    )

    lfr.fit(y=y_train, X=X_train)

    y_pred_4 = lfr.predict(X=X_test)
    y_pred_5 = lfr.predict(X=X, y=y_train)
    y_pred_6 = lfr.predict(X=X)

    pass


if __name__ == "__main__":
    main()
