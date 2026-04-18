from sktime.forecasting.base import ForecastingHorizon

import pandasx as pdx
import statsforecastx.models as stf
from pandasx import xy_split
from stdlib import qname

TARGET = "import_kg"
NUMERICAL = ["crude_oil_price","sandp_500_us","sandp_sensex_india","shenzhen_index_china","nikkei_225_japan",
             "max_temperature","mean_temperature","min_temperature","vap_pressure","evaporation","rainy_days"]
IGNORE = ["prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country"]
GROUPS = ["country", "item"]
DATETIME = ["date", "%Y/%m/%d %H:%M:%S", "M"]
BINHOT = ["imp_month"]


def check_model(model, X_train, y_train, X_test, y_test):
    print(qname.qualified_type(model))
    n = len(y_test)
    fh = ForecastingHorizon(list(range(1, n + 1)))

    model.fit(y_train, X_train)

    y_pred = model.predict(fh=fh, X=X_test)


def main():
    df = pdx.read_data(
        "./data/vw_food_import_kg_train_test_area_skill_mini.csv",
        datetime=DATETIME,
        numerical=NUMERICAL,
        ignore=IGNORE,
        binhot=BINHOT,
        dropna=True
    )
    # dfd = pdx.groups_split(df, groups=GROUPS, drop=True)

    dfg = pdx.groups_select(df, groups=GROUPS, values=["ARGENTINA", "ANIMAL FEED"], drop=True)
    pdx.set_index(dfg, columns=["date"], inplace=True, drop=True)

    train, test = pdx.train_test_split(dfg, train_size=0.8)
    X_train, y_train, X_test, y_test = xy_split(train, test, target=TARGET)

    check_model(stf.AutoARIMA(), X_train, y_train, X_test, y_test)
    check_model(stf.ARIMA(), X_train, y_train, X_test, y_test)
    check_model(stf.AutoCES(), X_train, y_train, X_test, y_test)
    check_model(stf.AutoETS(), X_train, y_train, X_test, y_test)
    # check_model(stf.AutoMFLES(), X_train, y_train, X_test, y_test)
    check_model(stf.MFLES(), X_train, y_train, X_test, y_test)
    check_model(stf.AutoTBATS(), X_train, y_train, X_test, y_test)
    check_model(stf.AutoTheta(), X_train, y_train, X_test, y_test)

    pass



if __name__ == "__main__":
    main()