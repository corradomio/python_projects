import numpy as np
import pandasx as pdx
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from stdlib import qualified_name

from etime import LinearForecastRegressor
from etime.scikit_model import ScikitForecastRegressor


def main():
    # d1 = pdx.read_data('stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'))
    # d1 = pdx.dataframe_index(d1, index=['agency', 'sku', 'date'], inplace=True)
    # # df.to_period('M')

    df = pdx.read_data('stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'),
                       index=['agency', 'sku', 'date'],
                       ignore=['agency', 'sku', 'date', 'timeseries'])

    dfdict = pdx.dataframe_split_on_groups(df, groups=['agency', 'sku'])

    # df = pdx.read_data('stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'))
    # df = pdx.dataframe_index(df, index=['agency', 'sku', 'date'])
    # df = pdx.dataframe_ignore(df, ['agency', 'sku', 'date', 'timeseries'])

    X, y = pdx.xy_split(df, target='volume')
    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, train_size=0.8)

    # fh = ForecastingHorizon(np.arange(1, len(y_test)+1))
    # fh = ForecastingHorizon(np.arange(1, len(y_test)+1), freq='M')
    # rfh = np.array([[i for i in range(1, 13)] for j in range(350)]).reshape(-1)
    # fh = ForecastingHorizon(rfh)
    fh = ForecastingHorizon(np.arange(1, 11))

    skr = ScikitForecastRegressor(
        class_name=qualified_name(LinearRegression),
        window_length=1,
        strategy='recursive'
    )
    skr.fit(X=X_train, y=y_train)

    y_pred_1 = skr.predict(X=X_test, fh=fh)

    lfr = LinearForecastRegressor(
        class_name=qualified_name(LinearRegression),
        lag=dict(
            length=1,
            current=False
        )
    )
    lfr.fit(X=X_train, y=y_train)

    y_pred_2 = lfr.predict(X=X_test, fh=fh)

    pass


if __name__ == "__main__":
    main()
