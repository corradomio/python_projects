import pandas as pd
import pandasx as pdx
from stdlib.is_instance import is_instance
import pandasx.is_instance



def main():
    df = pdx.read_data(
        "./vw_food_import_kg_train_test_area_skill_mini.csv",
        ignore=["prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country"]
    )

    print(is_instance(df, pd.DataFrame[
        [
        "item","country","imp_month","date","import_kg",
         # "prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country",
         "crude_oil_price","sandp_500_us","sandp_sensex_india","shenzhen_index_china","nikkei_225_japan",
         "max_temperature","mean_temperature","min_temperature","vap_pressure","evaporation","rainy_days"
        ]
    ]))


    print(is_instance(df, pd.DataFrame[
        {
        "item": str,"country": str,"imp_month": None,"date": None,"import_kg": None,
         # "prod_kg","avg_retail_price_src_country","producer_price_tonne_src_country",
         "crude_oil_price": None,"sandp_500_us": None,"sandp_sensex_india": None,"shenzhen_index_china": None,"nikkei_225_japan": None,
         "max_temperature": None,"mean_temperature": None,"min_temperature": None,"vap_pressure": None,"evaporation": None,"rainy_days": None
        }
    ]))
    pass


if __name__ == "__main__":
    main()
