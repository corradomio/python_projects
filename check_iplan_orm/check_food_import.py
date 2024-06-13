import logging.config

import pandasx as pdx
from iplan.om import IPlanObjectModel, TimeSeriesFocussed
from stdlib.jsonx import load
from commons import *


def main():
    datasource_dict = load('datasource_local.json')
    ipom = IPlanObjectModel(datasource_dict)

    df_train = pdx.read_data(
        "data_test/vw_food_import_kg_train_test_area_skill.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        na_values=['(null)'],
        ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )

    df_pred = pdx.read_data(
        "data_test/vw_food_import_kg_pred_area_skill.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        na_values=['(null)'],
        ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )

    with ipom.connect():
        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES).using_plan(PLAN_NAME, DATA_MASTER)
        assert ts.exists()

        print(ts.measures)

        ts.test().delete()
        ts.predicted().delete()
        print("ok")

        ts.train().delete()
        ts.train().save(df_train)

        ts.predict().delete()
        ts.predict().save(df_pred)

        dst = ts.train().select(area='ARGENTINA', skill='ANIMAL FEED')
        dsp = ts.predict().select(area='ARGENTINA', skill='ANIMAL FEED')

        print("done")
    return


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
