import datetime
import logging.config

import pandasx as pdx
from commons import *
from iplan.om import IPlanObjectModel, TimeSeriesFocussed, PredictionPlan
from stdlib.jsonx import load
from datetime import datetime


def main():
    datasource_dict = load('datasource_local.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect():
        assert ipom.hierachies().area_hierarchy(AREA_HIERARCHY).exists()
        assert ipom.hierachies().skill_hierarchy(SKILL_HIERARCHY).exists()
        assert ipom.data_models().data_model(DATA_MODEL).exists()
        assert ipom.data_masters().data_master(DATA_MASTER).exists()

        ipom.plans().plan(PLAN_NAME).delete()

        plan = ipom.plans().plan(PLAN_NAME, DATA_MASTER)
        if not plan.exists():
            plan.create(start_date=datetime.today())

        apmap = plan.area_plan_map
        pass


def main1():
    # datasource_dict = load('datasource_remote.json')
    datasource_dict = load('datasource_local.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect():
        plan1: PredictionPlan = ipom.plans().plan(PLAN_NAME, DATA_MASTER)
        print(plan1.exists())
        if not plan1.exists():
            plan1.create(start_date=datetime.datetime.today())

        apmap1 = plan1.area_plan_map

        plan2 = ipom.plans().plan(13357333)
        apmap2 = plan2.area_plan_map
        pass
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
