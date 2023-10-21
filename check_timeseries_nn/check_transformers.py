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
    # df = pdx.read_data(f"{DATA_DIR}/vw_food_import_train_test_newfeatures.csv",
    #                    datetime=DATETIME,
    #                    ignore=GROUPS + DATETIME[0:1] + [
    #                        "imp_month",
    #                        "prod_kg",
    #                        "avg_retail_price_src_country",
    #                        "producer_price_tonne_src_country",
    #                        "min_temperature",
    #                        "max_temperature",
    #
    #                        # "crude_oil_price",
    #                        # "sandp_500_us",
    #                        # "sandp_sensex_india",
    #                        # "shenzhen_index_china",
    #                        # "nikkei_225_japan",
    #
    #                        # "mean_temperature",
    #                        "vap_pressure",
    #                        "evaporation",
    #                        "rainy_days",
    #                    ],
    #                    onehot=["imp_month"],
    #                    dropna=True,
    #                    na_values=['(null)'],
    #                    index=GROUPS + DATETIME[0:1]
    #                    )

    df = pdx.read_data(f"{DATA_DIR}/stallion.csv",
                       datetime=['date', '%Y-%m-%d', 'M'],
                       index=['agency', 'sku', 'date'],
                       ignore=['timeseries', 'agency', 'sku', 'date'] + [
                           'industry_volume', 'soda-volume'
                       ],
                       binary=["easter_day",
                               "good_friday",
                               "new_year",
                               "christmas",
                               "labor_day",
                               "independence_day",
                               "revolution_day_memorial",
                               "regional_games",
                               "fifa_u_17_world_cup",
                               "football_gold_cup",
                               "beer_capital",
                               "music_fest"
                               ]
                       )

    dt = pdx.DTypeEncoder(dtype=float)
    X = dt.fit_transform(df).sort_index(axis=0)
    qs = pdx.MinMaxScaler()
    Xt = qs.fit_transform(X)
    Xo = qs.inverse_transform(Xt)
    Y = Xo.sort_index(axis=0)

    # ble = pdx.BinaryLabelsEncoder(columns=["new_year", "christmas"])
    # Xt = ble.fit_transform(X)
    #
    # dtyp = pdx.DTypeEncoder(columns=["labor_day", "independence_day"], dtype=float)
    # Xt = dtyp.fit_transform(X)

    pass


if __name__ == "__main__":
    main()
