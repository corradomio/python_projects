import logging.config

import pandasx as pdx
from commons import *
from iplan.om import IPlanObjectModel, TimeSeriesFocussed
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
        assert ipom.plans().plan(PLAN_NAME, DATA_MASTER).exists()

        data_model = ipom.data_models().data_model(DATA_MODEL)

        print(data_model.measures)

        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES)
        if not ts.exists():
            ts.create(targets=TS_TARGETS, inputs=TS_INPUTS, data_master=DATA_MASTER)

        print(ts.parameters)
        print(ts.measures)

        ts: TimeSeriesFocussed = ipom.time_series().focussed(TIME_SERIES).using_plan(PLAN_NAME, DATA_MASTER)
        # ts = ipom.time_series().focussed(TIME_SERIES).using_plan(13357333)
        pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
