import logging.config

import pandas as pd
import pandasx as pdx
from commons import *
from iplan.om import IPlanObjectModel, TimeSeriesFocussed
from stdlib.jsonx import load


def check_train(ipom: IPlanObjectModel):

    # area_id = 953
    # skill_id = [993]
    # plan_id = 13356658
    # time_series_id = "68"  # ipr_conf_master == time_series
    # data_master_id = "48"
    # from_date = pdx.to_datetime("Apr 29, 2024")

    time_series_id = TIME_SERIES
    data_master_id = DATA_MASTER
    area_id = "ARGENTINA"
    skill_id = "ANIMAL FEED"

    with ipom.connect():
        ts = ipom.time_series().focussed(time_series_id).using_data_master(data_master_id)
        area_hierarchy = ts.area_hierarchy
        skill_hierarchy = ts.skill_hierarchy
        data_master = ts.data_master

        df_train = ts.train().select(new_format=False)
        df_train = ts.train().select(area=area_id, skill=skill_id)
        pass
    pass


def check_predict(ipom: IPlanObjectModel):

    area_id = 953
    skill_id = [993]
    plan_id = 13356658
    time_series_id = "68"  # ipr_conf_master == time_series
    from_date = pdx.to_datetime("Apr 29, 2024")

    # time_series_id = TIME_SERIES
    # plan_id = PLAN_NAME
    # area_id = "ARGENTINA"
    # skill_id = "ANIMAL FEED"
    # from_date = None

    with ipom.connect():
        ts = ipom.time_series().focussed(time_series_id).using_plan(plan_id)
        area_hierarchy = ts.area_hierarchy
        skill_hierarchy = ts.skill_hierarchy
        data_master = ts.data_master
        measures = ts.measures
        plan = ts.plan

        start_date, end_date = ts.plan.date_range

        # df_train = ts.train().select(end_date=end_date, area=area_id, skill=skill_id, new_format=False)
        df_train = ts.train().select(new_format=False, use_plan=False)
        df_train = ts.train().select(new_format=False, use_plan=False, end_date=from_date)
        df_pred = ts.predict().select(area=area_id, skill=skill_id, new_format=False)
        pass
    pass


def main():
    datasource_dict = load('datasource_remote.json')
    ipom = IPlanObjectModel(datasource_dict)

    # check_train(ipom)
    check_predict(ipom)
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
