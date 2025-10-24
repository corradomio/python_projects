import pandasx as pdx

def main():
    df = pdx.read_data(
        "data/vw_food_import_kg_train_test_area_skill_mini.csv",
        numerical=["import_kg", "prod_kg", "avg_retail_price_src_country", "producer_price_tonne_src_country",
                 "crude_oil_price", "sandp_500_us", "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan",
                 "max_temperature", "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"],
        # datetime=("date", "%Y/%m/%d %H:%M:%S", "M"),
        ignore=["prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country", "date"],
        datetime=("date", "%Y/%m/%d %H:%M:%S"),
        onehot=["imp_month"],
        # datetime_index="date",
        na_values=["(null)"]
    )

    dfdict = pdx.groups_split(df, groups=["country", "item"])
    for g in dfdict:
        dfg = dfdict[g]

        X, y = pdx.xy_split(dfg, target="import_kg")

        xscaler = pdx.StandardScaler(feature_range=(1,2))
        yscaler = pdx.StandardScaler(feature_range=(1,2))
        # xscaler = pdx.LinearMinMaxScaler(feature_range=(-1,1))
        # yscaler = pdx.LinearMinMaxScaler(feature_range=(-1,1))

        X_scaled = xscaler.fit_transform(X)
        y_scaled = yscaler.fit_transform(y)
        y_restore = yscaler.inverse_transform(y_scaled)

        pass

    pass

    # df.info()
    #
    # dfg = dict(iter(df.groupby(by=["item","country"])))
    #
    # onehot = pdx.OneHotEncoder(columns=["imp_month"])
    #
    # dfoh = onehot.fit_transform(df)
    #
    # dfb = onehot.inverse_transform(dfoh)

    pass



if __name__ == "__main__":
    main()