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
        categorical=['imp_month'],
        na_values=['(null)'],
        ignore=['prod_kg', 'avg_retail_price_src_country', 'producer_price_tonne_src_country'],
    )

    countries = df_train['area'].unique().tolist()
    foods = df_train['skill'].unique().tolist()

    with ipom.connect():
        # ipom.time_series().focussed(TIME_SERIES).delete()
        # ipom.data_masters().data_master(DATA_MASTER).delete()
        # ipom.data_models().data_model(DATA_MODEL).delete()

        # h = ipom.hierachies().skill_hierarchy(AREA_HIERARCHY).delete()
        # ah: AttributeHierarchy = ipom.hierachies().area_hierarchy(AREA_HIERARCHY).delete()
        # sh: AttributeHierarchy = ipom.hierachies().skill_hierarchy(SKILL_HIERARCHY).delete()
        # ah.create({'WORLD': countries})
        # sh.create({'FEEDS': foods})

        # dmodel = ipom.data_models().data_model(DATA_MODEL) \
        #     .create(
        #         targets='import_kg',
        #         inputs=["prod_kg", "avg_retail_price_src_country",
        #                 "producer_price_tonne_src_country", "crude_oil_price", "sandp_500_us",
        #                 "sandp_sensex_india", "shenzhen_index_china", "nikkei_225_japan", "max_temperature",
        #                 "mean_temperature", "min_temperature", "vap_pressure", "evaporation", "rainy_days"],
        #         update=False
        #     )

        # dmaster = ipom.data_masters().data_master(DATA_MASTER) \
        #     .create(
        #         data_model=DATA_MODEL,
        #         area_hierarchy=AREA_HIERARCHY,
        #         skill_hierarchy=SKILL_HIERARCHY,
        #         period_hierarchy='month',
        #         periods=12
        #     )

        # plan = ipom.plans().plan(PLAN_NAME, DATA_MASTER)
        # plan.delete()
        # plan.create(datetime.strptime('2024-01-01', '%Y-%m-%d'))
        # print(plan.exists())
        pass
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
