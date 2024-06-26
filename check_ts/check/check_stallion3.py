import pandasx as pdx
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from stdlib import qualified_name

from sktimex import LinearForecastRegressor
from sktimex.scikit_model import ScikitForecastRegressor


def main():
    # d1 = pdx.read_data('stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'))
    # d1 = pdx.dataframe_index(d1, index=['agency', 'sku', 'date'], inplace=True)
    # # df.to_period('M')

    df = pdx.read_data('../data/stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'), index='date')
    dfdict = pdx.dataframe_split_on_groups(df, groups=['agency', 'sku'])

    for ts in dfdict.keys():
        print(ts)

        df = dfdict[ts]
        df = pdx.dataframe_ignore(df, ['agency', 'sku', 'date', 'timeseries'])

        # df = pdx.read_data('stallion_all.csv', datetime=('date', '%Y-%m-%d', 'M'))
        # df = pdx.dataframe_index(df, index=['agency', 'sku', 'date'])
        # df = pdx.dataframe_ignore(df, ['agency', 'sku', 'date', 'timeseries'])

        X, y = pdx.xy_split(df, target='volume')
        X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, train_size=0.8)

        X.to_csv("X.csv")

        # fh = ForecastingHorizon(np.arange(1, len(y_test)+1))
        # fh = ForecastingHorizon(np.arange(1, len(y_test)+1), freq='M')
        # rfh = np.array([[i for i in range(1, 13)] for j in range(350)]).reshape(-1)
        # fh = ForecastingHorizon(rfh)
        # fh = ForecastingHorizon(np.arange(1, 11))
        fha = ForecastingHorizon(X_test.index, is_relative=False)
        fhr = fha.to_relative(y_train.index[-1])

        skr = ScikitForecastRegressor(
            class_name=qualified_name(LinearRegression),
            window_length=1,
            strategy='recursive'
        )
        skr.fit(X=X_train, y=y_train)
        y_pred_1 = skr.predict(X=X_test, fh=fhr)

        # fha = ForecastingHorizon(X_test.index, is_relative=False)
        # fhr = fha.to_relative(skr.cutoff)
        # fh1 = fhr.to_relative(skr.cutoff)
        # fh2 = ForecastingHorizon([1, 3, 5]).to_absolute(skr.cutoff)
        #
        # y_pred_11 = skr.predict(X=X_test, fh=ForecastingHorizon([1]))
        # y_pred_12 = skr.predict(X=X_test, fh=ForecastingHorizon([1, 2]))
        # y_pred_13 = skr.predict(X=X_test, fh=ForecastingHorizon([1, 2, 3]))
        # y_pred_14 = skr.predict(X=X_test, fh=ForecastingHorizon([1, 2, 3, 4]))
        # y_pred_15 = skr.predict(X=X_test, fh=ForecastingHorizon([1, 2, 3, 4, 5]))
        # y_pred_16 = skr.predict(X=X, fh=ForecastingHorizon([1, 2, 3, 4, 5]))
        #
        # y_pred_1b = skr.predict(X=X_test, fh=ForecastingHorizon([1, 3, 5]))
        # y_pred_1c = skr.predict(X=X, fh=fh2)
        #
        # y_pred_a1 = skr.predict(X=X_test, fh=fha)
        # y_pred_a2 = skr.predict(X=X, fh=fha)
        # y_pred_r1 = skr.predict(X=X_test, fh=fhr)
        # y_pred_r2 = skr.predict(X=X, fh=fhr)
        # y_pred_r3 = skr.predict(X=X, fh=ForecastingHorizon(list(range(1, 10))))

        lfr = LinearForecastRegressor(
            class_name=qualified_name(LinearRegression),
            lag=dict(
                length=1,
                current=False
            )
        )
        lfr.fit(X=X_train, y=y_train)
        y_pred_2 = skr.predict(X=X_test, fh=fhr)

        pass
    # end


if __name__ == "__main__":
    main()
