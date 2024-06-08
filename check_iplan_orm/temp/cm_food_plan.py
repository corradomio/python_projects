import logging.config
from datetime import datetime

import pandasx as pdx
from iplan.om import IPlanObjectModel, IPredictionPlan, IDataMaster
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
    from typing import Literal
    PERIOD_HIERARCHY: Literal['day', 'week', 'month'] = 'month'
    PERIODS = 12

    with ipom.connect():

        # dm: IDataMaster = ipom.data_masters().data_master(DATA_MASTER)
        # print(dm.exists())
        # dm.delete()
        # dm.create(DATA_MODEL, AREA_HIERARCHY, SKILL_HIERARCHY,
        #           PERIOD_HIERARCHY, PERIODS, update=False)

        plan: IPredictionPlan = ipom.plans().plan(PLAN_NAME, DATA_MASTER)
        print(plan.exists())
        plan.delete()
        plan.create(datetime.strptime('2024-01-01', '%Y-%m-%d'))
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
