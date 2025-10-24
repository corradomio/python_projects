import pandasx as pdx
from sktimex.forecasting import create_forecaster
from sktime.forecasting.base import ForecastingHorizon


def main():
    df_all = pdx.read_data(
        "./data/vw_food_import_kg_train_test_area_skill.csv",
        datetime=("date", "%Y/%m/%d %H:%M:%S"),
        onehot=["imp_month"],
        numerical=[
            "import_kg",
            "crude_oil_price", "sandp_500_us", "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan",
            "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"
        ],
        ignore=[
            "prod_kg",
            "avg_retail_price_src_country",
            "producer_price_tonne_src_country",
            "max_temperature",
            "date"
        ],
        datetime_index=["date"],
        na_values=["(null)"]
    )

    df = pdx.groups_select(df_all, dict(
        country="ALGERIA",
        item="DATES"
    ), drop=True)

    df_train, df_test = pdx.train_test_split(df, test_size=12)
    X_train, y_train, X_test, y_true = pdx.xy_split(df_train, df_test, target="import_kg")

    f1 = create_forecaster("lin", {
        "class": "sklearn.linear_model.LinearRegression",
        # "window_length": 12,
        # "prediction_length": 1
        "lags": 12,
        "tlags": 1
    })

    f1.fit(y=y_train, X=X_train)
    fh = ForecastingHorizon(y_true.index)
    y_pred_1 = f1.predict(fh=fh, X=X_test)

    # f2 = create_forecaster("lin", {
    #     "class": "sklearn.linear_model.LinearRegression",
    #     "window_length": 12,
    #     "prediction_length": 1
    #     # "lags": 12,
    #     # "tlags": 1
    # })
    #
    # f2.fit(y=y_train, X=X_train)
    # fh = ForecastingHorizon(y_true.index)
    # y_pred_2 = f2.predict(fh=fh, X=X_test)

    # f1.update(None, None, update_params=False)

    f1.update(y=y_train, X=X_train, update_params=False)
    fh = ForecastingHorizon(y_true.index, freq='MS')
    y_pred_2 = f1.predict(fh=fh, X=X_test)


    pass




if __name__ == "__main__":
    main()
