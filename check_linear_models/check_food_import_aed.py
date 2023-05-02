import pandas as pd
import pandasx as pdx


def main():
    df = pdx.read_data("vw_food_import_aed_pred.csv",
                       datetime=("imp_date", "%Y-%m-%d"),
                       categorical=["imp_month"],
                       index=["item_country", "imp_date"],
                       ignore=[""])
    print(df.info())
    pass


if __name__ == "__main__":
    main()
