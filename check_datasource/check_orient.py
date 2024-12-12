import json
from typing import cast, Literal

import pandas as pd
import pandasx as pdx
import stdlib.logging as logging
import stdlib.jsonx as json


def main():
    df: pd.DataFrame = pdx.read_data(
        "./vw_food_import_kg_train_test_area_skill_mini.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        # categorical=['imp_month'],
        na_values=["(null)"]
    )
    # jdata = pdx.to_json(df, orient="columns")
    # jdata = pdx.to_json(df, orient="flat_columns")

    for orient in ["dict", "list", "series", "split", "tight", "records", "index"]:
        data = df.to_dict(orient=cast(Literal[
                                          "dict", "list", "series", "split", "tight", "index"
                                      ], orient))

        json.dump(data, f"formats/dict-{orient}.json")
        pass

    orient = "list"
    path = f"formats/{orient}.json"
    pdx.write_data(df, path,
                   orient=orient,
                   index=4,
                   datetime='%Y/%m/%d %H:%M:%S'
                   )

    dff = pdx.read_data(path, orient=orient, datetime=('date', '%Y/%m/%d %H:%M:%S'))

    for orient in ["split", "records", "index", "table", "columns", "values", "list"]:
        print(orient)
        # pdx.to_json(df, f"formats/{orient}.json",
        #             orient=orient,
        #             indent=4,
        #             datetime='%Y/%m/%d %H:%M:%S'
        # )
        pdx.write_data(df, f"formats/{orient}.json",
                       orient=orient,
                       index=4,
                       datetime='%Y/%m/%d %H:%M:%S'
       )

    print(df.head())
    pass




if __name__ == '__main__':
    logging.config.fileConfig('logging_config.ini')
    logging.getLogger('root').info('Logging initialized')
    main()
