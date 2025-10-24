import pandas as pd
import pandasx as pdx


def main():
    df: pd.DataFrame = pdx.read_data(
        "vw_food_import_kg_train_test_area_skill.csv",
        na_values=["(null)"],
        categorical=["imp_month"],
        datetime=("date", "%Y/%m/%d %H:%M:%S"),
        ignore=["prod_kg", "avg_retail_price_src_country", "producer_price_tonne_src_country"],
        numerical=[
            "crude_oil_price", "nikkei_225_japan", "sandp_500_us", "sandp_sensex_india", "shenzhen_index_china",
            "max_temperature", "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"
        ]
    )

    df.sort_values(by=["item"], ascending=True, inplace=True)

    print(type(df.index))
    pass


if __name__ == "__main__":
    main()
