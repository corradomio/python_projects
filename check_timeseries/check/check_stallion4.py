import numpy as np
import pandasx as pdx
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from stdlib import qualified_name

from etime import LinearForecastRegressor
from etime.scikit_model import ScikitForecastRegressor


def main():
    df = pdx.read_data('../stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'),
                       index=['agency', 'sku', 'date'],
                       ignore=['agency', 'sku', 'date', 'timeseries'])

    X, y = pdx.xy_split(df, target='volume')

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, train_size=0.8)
    fh = ForecastingHorizon(np.arange(1, 11))

    # --

    # skr = ScikitForecastRegressor(
    #     class_name=qualified_name(LinearRegression),
    #     window_length=1,
    #     strategy='recursive'
    # )
    # skr.fit(X=X_train, y=y_train)
    #
    # y_pred_1 = skr.predict(fh=fh, X=X_test)
    # yp_1 = y_pred_1.loc[('Agency_22', 'SKU_01')]

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


if __name__ == "__main__":
    main()
