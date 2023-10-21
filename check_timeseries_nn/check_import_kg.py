import logging
import warnings

import pandasx as pdx

# hide warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("root")

DATA_DIR = "./data"

DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
TARGET_2 = "import_kg_2"
GROUPS = ['item_country']


def main():
    df = pdx.read_data(f"{DATA_DIR}/vw_food_import_train_test_newfeatures.csv",
                       datetime=DATETIME,
                       ignore=GROUPS + DATETIME[0:1] + [
                           "imp_month",
                           "prod_kg",
                           "avg_retail_price_src_country",
                           "producer_price_tonne_src_country",
                           "min_temperature",
                           "max_temperature",

                           # "crude_oil_price",
                           # "sandp_500_us",
                           # "sandp_sensex_india",
                           # "shenzhen_index_china",
                           # "nikkei_225_japan",

                           # "mean_temperature",
                           "vap_pressure",
                           "evaporation",
                           "rainy_days",
                       ],
                       onehot=["imp_month"],
                       dropna=True,
                       na_values=['(null)'],
                       index=GROUPS + DATETIME[0:1]
                       )

    # glist = pdx.groups_list(df)

    dfdict = pdx.groups_split(df)
    df = pdx.groups_merge(dfdict)

    scaler = pdx.StandardScaler()
    scaler.fit_transform(df)


    train, test = pdx.train_test_split(df, test_size=12)


    pass

if __name__ == "__main__":
    main()
