import logging.config
from iplan.om import IPlanObjectModel, IPredictMasterFocussed, IPredictDetailFocussed, IDataValuesMaster
from sqlalchemy import URL
from stdlib import lrange
from stdlib.jsonx import load
from datetime import datetime


def main():
    datasource_dict = load('datasource.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect():
        pp = ipom.prediction_plans()

        # print(pp.delete_plan('Check_My_Plan'))
        # pp.create_plan('Check_My_Plan', data_master=48, start_date=datetime(2024, 4, 1))

        pf = ipom.predict_focussed(68)
        print("data_master_id", pf.data_master.id)
        print("data_model_id", pf.data_model.id)
        print("area_feature_ids", pf.area_hierarchy.feature_ids())
        print(pf.select_data_master_ids())
        print(pf._select_data_values_master_ids())

        train_data = pf.select_train_data(new_format=True)
        prediction_data = pf.select_prediction_data(new_format=True)
    pass
# end


def main5():
    datasource_dict = load('datasource.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect():

        pp = ipom.prediction_plans()

        print(pp.exists_plan('Check_My_Plan', 7))
        print(pp.exists_plan(datetime(2024, 5, 1, 0, 0, 0), 7))

        # pp.delete_plan("Check_My_Plan")

        pp.create_plan(name='Check_My_Plan',
                       start_date=datetime(2024, 5, 1, 0, 0, 0),
                       # end_date=datetime(224, 5, 31, 23,59,59),
                       data_master=7,
                       force=False)

        # date_interval = pp.select_date_interval("13339197")
        # date_interval = pp.select_date_interval("Autoupdate_Plan_sikhar_2023-02-26 0:00:00")
        date_interval = pp.select_date_interval('Check_My_Plan', data_master_id=7)

        date_interval = pp.select_date_interval(13356946, data_master_id=7)
        date_interval = pp.select_date_interval(None, data_master_id=7, area_feature_ids=lrange(203, 255))

        pf = ipom.predict_focussed(68)
        pf.select_prediction_data(date_interval, new_format=False)

        prdct_data = pf.select_prediction_data(plan_id='Check_My_Plan', new_format=True)

        pass
    # end
    print("done")


def main4():
    datasource_dict = load('datasource.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect() as conn:
        dvm = ipom.data_values_master(13355950)

        ah = dvm.area_hierarchy

        pass
    # end
    print("done")


def main3():
    datasource_dict = load('datasource.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect() as conn:
        pf = ipom.predict_focussed(68)
        train_data = pf.select_train_data(new_format=True)
        prdct_data = pf.select_prediction_data(None, new_format=True)

        pass
    # end
    print("done")


def main2():
    datasource_dict = load('datasource.json')
    ipom = IPlanObjectModel(datasource_dict)

    try:
        ipom.connect()

        pf = ipom.predict_focussed('IXD_NEW_IPREDICT')

        train_data = pf.select_train_data(new_format=True)

        print(train_data)
        pass
    finally:
        ipom.disconnect()


def main1():
    datasource_dict = load('datasource.json')
    ipom = IPlanObjectModel(datasource_dict)

    try:
        ipom.connect()

        pf = ipom.predict_focussed('CM iPredict Master v2')
        # print(pf.parameters)
        # print(pf.input_target_parameters)
        # print(pf.measures)
        # print(pf.area_hierarchy)
        # print(pf.skill_hierarchy)
        # print(pf.data_master)
        # print(pf.data_master.area_hierarchy)
        # print(pf.data_master.skill_hierarchy)
        # print(pf.data_master.data_model)
        # print(*pf.input_target_measure_ids)

        print(pf.select_train_data())

        # data_model = ipom.data_model('IXD Model Master')
        # measures = data_model.details()
        #
        # for measure in measures:
        #     print(measure)

        # data_master = ipom.data_master('IXD NEW')
        # print(data_master.data_model)
        # print(data_master.area_hierarchy)
        # print(data_master.skill_hierarchy)
        #
        # measures = data_master.data_model.measures()
        #
        # for measure in measures:
        #     print(measure)

        # pmf: IPredictMasterFocussed = ipom.predict_focussed('CM iPredict Master v2')
        # inputs, targets = pmf.inputs_targets()

        # area = ipom.area_hierarchy('Countries')
        # print(area.name)
        # print(area.description)
        #
        # root = area.tree()
        # print(root.name, ":", root.description)
        #
        # skill = ipom.skill_hierarchy('Foods')
        # print(skill.name)
        # print(skill.description)
        #
        # root = skill.tree()
        # print(root.name, ":", root.description)
        print("done")

    finally:
        ipom.disconnect()

    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
