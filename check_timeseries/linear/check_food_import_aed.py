import pandasx as pdx
from sklearn.linear_model import LinearRegression
from sktime.forecasting.base import ForecastingHorizon
from stdlib import qualified_name

from sktimex import LinearForecastRegressor


def main_train():
    df = pdx.read_data("vw_food_import_aed_train_test.csv",
                       datetime=("imp_date", "%Y-%m-%d"),
                       categorical=["imp_month"],
                       index=["item_country", "imp_date"],
                       ignore_unnamed=True,
                       dropna=True)
    df1 = df.dropna(how='any')
    print(df.info())
    pass


def main_pred():
    dftt = pdx.read_data(
        "vw_food_import_aed_train_test.csv",
        datetime=("imp_date", "%Y-%m-%d", "M"),
        onehot=["imp_month"],
        index=["item_country", "imp_date"],
        ignore=["item_country", "imp_month", "imp_date"],
        ignore_unnamed=True,
        dropna=False)

    dftt = dftt.loc[('ANIMAL FEED~ARGENTINA',)]
    X, y = pdx.xy_split(dftt, target='import_aed')
    train, test = pdx.train_test_split(dftt, train_size=0.8)

    X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target='import_aed')
    fh = ForecastingHorizon(y_test.index, is_relative=False)

    lfr = LinearForecastRegressor(
        estimator=qualified_name(LinearRegression),
        lags=dict(
            length=5,
            current=False
        )
    )

    lfr.fit(y=y_train, X=X_train)
    y_pred_1 = lfr.predict(fh=fh, X=X_test)
    y_pred_2 = lfr.predict(fh=fh, X=X)
    y_pred_3 = lfr.predict(fh=fh, X=X)

    # --

    dfp = pdx.read_data(
        "../data/vw_food_import_aed_pred.csv",
        datetime=("imp_date", "%Y-%m-%d", "M"),
        onehot=["imp_month"],
        index=["item_country", "imp_date"],
        ignore=["item_country", "imp_month", "imp_date", "Unnamed: 0"],
        dropna=False)

    dfp = dfp.loc[('ANIMAL FEED~ARGENTINA',)]
    X, y = pdx.xy_split(dfp, target='import_aed')
    y_valid, y_nan = pdx.nan_split(y, target='import_aed')
    fh = ForecastingHorizon(y_nan.index, is_relative=False)

    y_pred_4 = lfr.predict(fh=fh, X=X, y=y_valid)

    pass


if __name__ == "__main__":
    main_pred()
