import pandasx as pdx
import pandas as pd
from stdlib import jsonx


def main():
    df: pd.DataFrame = pdx.read_data("data/jena_climate_2009_2016.csv")

    for orient in ["split", "records", "index", "table", "columns", "values", "pivot"]:
        # df.to_json(f"test-{orient}.json", orient=orient)
        jdata = pdx.to_json(df, f"test-{orient}.json", orient=orient)


if __name__ == "__main__":
    main()
