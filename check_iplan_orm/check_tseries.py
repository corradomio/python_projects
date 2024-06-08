import logging.config

import pandasx as pdx
from commons import *
from iplan.om import IPlanObjectModel, TimeSeriesFocussed
from stdlib.jsonx import load


def main():
    # datasource_dict = load('datasource.json')
    datasource_dict = load('datasource_localhost.json')
    ipom = IPlanObjectModel(datasource_dict)

    df_train = pdx.read_data(
        "data_test/vw_food_import_kg_train_test_area_skill.csv",
        datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
        # categorical=['imp_month'],
        na_values=['(null)'],
        ignore=['prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country', 'imp_month'],
    )

    with ipom.connect():

        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES)
        print(TIME_SERIES, ts.exists())
        # ts.delete()
        # ts.create(
        #     targets=TS_TARGETS,
        #     populate=None,
        #     inputs=TS_INPUTS,
        #     data_master=DATA_MASTER
        # )
        # ts.set_plan(PLAN_NAME, DATA_MASTER)
        ts.set_plan(PLAN_NAME)

        print(ts.parameters)
        print(ts.measures)
        print(ts.measure_ids(with_name=False))
        print(ts.measure_ids(with_name=True))

    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
