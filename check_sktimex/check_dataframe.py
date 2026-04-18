from typing import Optional

import pandas as pd
from neuralforecast.tsdataset import TimeSeriesDataset
from sktime.forecasting.base import ForecastingHorizon

import sktimex
import sktimexnn
import statsforecastx.models as stf
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


def to_tsds(y: pd.Series, X: Optional[pd.DataFrame], ignores_exogenous_X) -> TimeSeriesDataset:
    df = None

    if ignores_exogenous_X:
        X = None

    if y is not None:
        time = y.index.to_series()
        df = pd.DataFrame({
            "unique_id": 1,
            "ds": time,
            "y": y
        })

        if X is not None:
            df = pd.concat([df, X], axis=1)

    elif X is not None:
        df = X.copy()
        time = df.index.to_series()
        df["ds"] = time
        df["unique_id"] = 1

    assert df is not None

    df.reset_index(drop=True, inplace=True)
    return TimeSeriesDataset.from_df(df)



def main():
    df = pdx.read_data(
        "./data/vw_food_import_kg_train_test_area_skill_mini.csv",
        datetime=DATETIME,
        numerical=NUMERICAL,
        ignore=IGNORE + ["date"],
        binhot=BINHOT,
        datetime_index=["date"],
        dropna=True
    )
    # dfd = pdx.groups_split(df, groups=GROUPS, drop=True)

    dfg = pdx.groups_select(df, groups=GROUPS, values=["ARGENTINA", "ANIMAL FEED"], drop=True)

    X, y = xy_split(dfg, target=TARGET)

    tsds = to_tsds(y, X, ignores_exogenous_X=False)

    pass



if __name__ == "__main__":
    main()