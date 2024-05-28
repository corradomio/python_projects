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
        ipom.delete_skill_hierarchy(AREA_HIERARCHY)
        ipom.delete_area_hierarchy(AREA_HIERARCHY)
        ipom.delete_skill_hierarchy(SKILL_HIERARCHY)
        ipom.create_area_hierarchy(AREA_HIERARCHY, {'WORLD': countries})
        ipom.create_skill_hierarchy(SKILL_HIERARCHY, {'FEEDS': foods})

        AreaHierarchy = ipom.create_area_hierarchy(AREA_HIERARCHY)

    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
