import logging.config
from datetime import datetime

import pandasx as pdx
from iplan.om import IPlanObjectModel, IPredictTimeSeries
from stdlib.jsonx import load


def main():
    # datasource_dict = load('datasource.json')
    datasource_dict = load('datasource_localhost.json')
    ipom = IPlanObjectModel(datasource_dict)

    PLAN_NAME = 'Plan Food Import'
    DATA_MODEL = "CM Data Model Food Import"
    DATA_MASTER = "CM Data Master Food Import"
    AREA_HIERARCHY = "ImportCountries"
    SKILL_HIERARCHY = "ImportFoods"
    TIME_SERIES = "CM Time Series Food Import"

    with ipom.connect():
        # data_model = ipom.data_model('CM Data Model Master')
        # for m in data_model.measures():
        #     print(m)

        area_hierarchy = ipom.area_hierarchy(AREA_HIERARCHY)
        skill_hierarchy = ipom.skill_hierarchy(SKILL_HIERARCHY)

        # ipom.delete_data_model(DATA_MODEL)
        data_model = ipom.create_data_model(
            DATA_MODEL,
            targets='import_kg',
            inputs=["prod_kg", "avg_retail_price_src_country",
                    "producer_price_tonne_src_country", "crude_oil_price", "sandp_500_us",
                    "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan", "max_temperature",
                    "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"],
            update=False)
        # data_model = ipom.data_model(DATA_MODEL)

        # ipom.delete_data_master(DATA_MASTER)
        data_master = ipom.create_data_master(
            DATA_MASTER,
            data_model=DATA_MODEL,
            area_hierarchy=AREA_HIERARCHY,
            skill_hierarchy=SKILL_HIERARCHY,
            period_hierarchy='month',
            periods=12,
            update=False
        )
        # data_master = ipom.data_master(DATA_MASTER)

        df_train = pdx.read_data(
            "data_test/vw_food_import_kg_train_test_area_skill.csv",
            datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
            na_values=['(null)'],
            ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
        )

        # 'area','skill','crude_oil_price','date','evaporation','imp_month','import_kg','max_temperature',
        # 'mean_temperature','min_temperature','nikkei_225_japan','rainy_days','sandp_500_us','sandp_sensex_india',
        # 'shenzhen_index_china','vap_pressure'
        df_pred = pdx.read_data(
            "data_test/vw_food_import_kg_pred_area_skill.csv",
            datetime=('date', '%Y/%m/%d %H:%M:%S', 'M'),
            na_values=['(null)'],
            ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
        )

        ipom.delete_time_series_focussed(TIME_SERIES)
        ts: IPredictTimeSeries = ipom.create_time_series_focussed(
            TIME_SERIES,
            targets='import_kg',
            inputs=["crude_oil_price", "sandp_500_us",
                    "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan", "max_temperature",
                    "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"],
            data_master=DATA_MASTER,
            update=False)
        # ts: IPredictTimeSeries = ipom.time_series_focussed(TIME_SERIES, DATA_MASTER)

        # print(ts.name)
        # print(ts.data_master.name)
        # print(ts.data_model.name)
        # print(ts.area_hierarchy.name)
        # print(ts.skill_hierarchy.name)

        # ipom.delete_prediction_plan(PLAN_NAME, DATA_MASTER)
        plan = ipom.create_prediction_plan(
            PLAN_NAME, DATA_MASTER,
            start_date=pdx.to_datetime(df_train['date'].max() + 1),
            update=False)
        # plan = ipom.prediction_plan(PLAN_NAME, DATA_MASTER)

        # -------------------------------------------------------------------

        ts.set_plan(PLAN_NAME)

        # ts.delete_train_data(plan=PLAN_NAME)
        # ts.delete_predict_data(plan=PLAN_NAME)
        #
        # ts.save_train_data(df_train, plan=PLAN_NAME)
        # ts.save_predict_data(df_pred, plan=PLAN_NAME)
        #
        # dft = ts.select_train_data(plan=PLAN_NAME)
        # dfp = ts.select_predict_data(plan=PLAN_NAME)

        ts.delete_train_data()
        ts.delete_predict_data()

        ts.save_train_data(df_train)
        ts.save_predict_data(df_pred)

        dft = ts.select_train_data()
        dfp = ts.select_predict_data()

    print("done")


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()