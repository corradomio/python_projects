from typing import Literal

import stdlib.jsonx as jsonx
import pandasx as pdx

Lit = Literal["split", "records", "index", "table", "columns", "values"]


def main():

    df = pdx.read_data(
        r"D:\Projects.ebtic\ipredict-server-v17\data_test\vw_food_import_kg_train_test_area_skill_mini.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        # categorical=['imp_month'],
        na_values=['(null)'],
        ignore=['prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )

    # df = pd.read_csv(r"D:\Projects.ebtic\ipredict-server-v17\data_test\vw_food_import_kg_train_test_area_skill_mini.csv")

    # for orient in ["split", "records", "index", "columns", "table", "values"]:
    #     # jdata = pdx.to_json(df, orient=cast(Lit, orient), date_format='iso' if orient in ['table'] else '%Y-%m')
    #     pdx.to_json(df, f"data-{orient}.json", orient=cast(Lit, orient))
    #     pass

    jsonx.dump({
        "status": df
    }, "data.json", orient="split", date_format="%Y/%d")


    pass




if __name__ == "__main__":
    main()
