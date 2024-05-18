import logging.config
from datetime import datetime

import pandasx as pdx
from iplan.om import IPlanObjectModel, IPredictTimeSeries
from stdlib.jsonx import load


def main():
    # datasource_dict = load('datasource.json')
    datasource_dict = load('datasource_localhost.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect():
        # ts: IPredictTimeSeries = ipom.time_series_focussed("68", data_master="48")
        ts: IPredictTimeSeries = ipom.time_series_focussed("CM Food Import", data_master="CM Data Master Food Import")

        # print(ts.name)
        # print(ts.data_master.name)
        # print(ts.data_model.name)
        # print(ts.area_hierarchy.name)
        # print(ts.skill_hierarchy.name)

        df_train = pdx.read_data(
            "data_test/vw_food_import_kg_train_test_area_skill.csv",
            datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
            categorical=['imp_month'],
            na_values=['(null)'],
            ignore=['prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
        )

        plan = ipom.prediction_plan('Food Import', 55)
        plan.create(datetime.now(), force=False)
        ts.delete_train_data(plan=plan.name)
        ts.save_train_data(df_train, plan=plan.name)

        df_pred = pdx.read_data(
            "data_test/vw_food_import_kg_pred_area_skill.csv",
            datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
            categorical=['imp_month'],
            na_values=['(null)'],
            ignore=['prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
        )

        plan = ipom.prediction_plan('Food Import', 55)
        plan.create(datetime.now(), force=False)

        ts.delete_predict_data(plan='Food Import')
        ts.save_predict_data(df_pred, plan='Food Import')

    print("done")


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
