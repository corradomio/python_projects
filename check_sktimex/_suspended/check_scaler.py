
import logging.config
import pandasx as pdx

DATASET = "./data/vw_food_import_kg_train_test_area_skill_mini.csv"
TARGET = "import_kg"
NUMERICAL = ["crude_oil_price","sandp_500_us","sandp_sensex_india","shenzhen_index_china","nikkei_225_japan",
             "max_temperature","mean_temperature","min_temperature","vap_pressure","evaporation","rainy_days"]
IGNORE = ["prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country"]


def main():
    print("dataframe")
    df = pdx.read_data(
        DATASET,
        numerical=NUMERICAL,
        ignore=IGNORE,
        datetime=("date", "%Y/%m/%d %H:%M:%S", "M"),
        onehot=["imp_month"]
    )

    print("loop")
    dfdict = pdx.groups_split(df, groups=["country", "item"])
    for g in dfdict:
        print("...", g)
        dfg = dfdict[g]

        dfd = dfg.drop(columns=["date"])

        X, y = pdx.xy_split(dfd, target=TARGET)

        xscaler = pdx.StandardScaler(columns=NUMERICAL)
        yscaler = pdx.StandardScaler(columns=TARGET)
        # xscaler = pdx.LinearMinMaxScaler(columns=NUMERICAL)
        # yscaler = pdx.LinearMinMaxScaler(columns=TARGET)

        X_scaled = xscaler.fit_transform(X)
        y_scaled = yscaler.fit_transform(y)

        X_restored = xscaler.inverse_transform(X_scaled)
        y_restored = yscaler.inverse_transform(y_scaled)
        pass

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
