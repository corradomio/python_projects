from typing import Optional, Union

from neuralforecast.tsdataset import TimeSeriesDataset
from sktime.forecasting.base import ForecastingHorizon
import neuralforecast.models as nfm
import neuralforecastx.models as nfxm
import neuralforecast.common._base_model as nfc
import pandas as pd
import pandasx as pdx

from pandasx import xy_split
from stdlib import qname

TARGET = "import_kg"
NUMERICAL = ["crude_oil_price","sandp_500_us","sandp_sensex_india","shenzhen_index_china","nikkei_225_japan",
             "max_temperature","mean_temperature","min_temperature","vap_pressure","evaporation","rainy_days"]
IGNORE = ["prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country"]
GROUPS = ["country", "item"]
DATETIME = ["date", "%Y/%m/%d %H:%M:%S", "M"]
BINHOT = ["imp_month"]


def to_tsds(y: pd.Series, X: Union[None, pd.DataFrame, list[pd.DataFrame]], ignores_exogenous_X) -> Optional[TimeSeriesDataset]:
    assert isinstance(y, (pd.Series, list))
    assert isinstance(X, (type(None), pd.DataFrame, list))
    index = None

    if ignores_exogenous_X:
        X = None

    if isinstance(y, list):
        y = pd.concat(y, axis=0)

    if isinstance(X, list):
        X = pd.concat(X, axis=0)

    if X is None:
        index = y.index
        df = pd.DataFrame({
            "ds": y.index.to_series(),
            "y": y.values,
            "unique_id": 1
        })
    else:
        index = y.index
        df = pd.DataFrame({
            "ds": index.to_series(),
            "y": y.values,
            "unique_id": 1
        })
        df = pd.concat([df, X], axis=1, ignore_index=False)
    # end

    if isinstance(df["ds"].dtype, pd.PeriodDtype):
        freq = index.freq
        df["ds"] = df["ds"].map(lambda t: t.to_timestamp(freq=freq))
    # end

    df.reset_index(drop=True, inplace=True)
    dataset, indices, dates, ds = TimeSeriesDataset.from_df(df)
    return dataset


def check_model(model, X_train, y_train, X_test, y_test):
    print(qname.qualified_type(model))
    n = len(y_test)
    fh = ForecastingHorizon(list(range(1, n + 1)))

    model.fit(y_train, X_train)

    y_pred = model.predict(fh=fh, X=X_test)

    return y_pred


def check_model_plain(model: nfc.BaseModel, X_train, y_train, X_test, y_test):
    n = len(y_test)

    tsds = to_tsds(y_train, X_train, False)

    model.fit(tsds)

    tsds_pred = to_tsds(y_train, [X_train, X_test], False)

    y_pred = model.predict(tsds_pred)
    # y_pred: np.ndarray[h,1]

    pass


def main():
    futr_df = pd.read_csv(
        'https://datasets-nixtla.s3.amazonaws.com/EPF_FR_BE_futr.csv',
        parse_dates=['ds'],
    )
    futr_df.head()

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

    # y_pred = check_model(nfx.Autoformer(24), X_train, y_train, X_test, y_test)

    # check_model_plain(
    #     nfm.Autoformer(
    #         1, 24,
    #         max_steps=100
    #     ),
    #     None, y_train, None, y_test
    # )

    # check_model_plain(
    #     nfm.TiDE(
    #         3, 24,
    #         hist_exog_list=list(X_train.columns),
    #         max_steps=100
    #     ),
    #     X_train, y_train, X_test, y_test
    # )

    check_model(
        nfxm.TiDE(
            24, h=1,
            max_steps=100
        ),
        X_train, y_train, X_test, y_test
    )
    pass



if __name__ == "__main__":
    main()