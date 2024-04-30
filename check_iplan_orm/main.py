import logging.config
from iplan.om import IPlanObjectModel, IPredictMasterFocussed, IPredictDetailFocussed, IDataValuesMaster
from sqlalchemy import URL
from stdlib.jsonx import load


# def main():
#     datasource_dict = load('datasource.json')
#     url = URL.create(**datasource_dict)
#
#     ipom = IPlanObjectModel(url)
#
#     with ipom.connect() as conn:
#
#         dvm: IDataValuesMaster = ipom.data_values_master("13355950")
#         start_date = dvm.start_date
#         end_date = dvm.end_date
#         area_feature = dvm.area_feature
#
#         prdct_data = dvm.select_prediction_data(new_format=True)
#         pass


def main():
    datasource_dict = load('datasource.json')
    url = URL.create(**datasource_dict)

    ipom = IPlanObjectModel(url)

    with ipom.connect() as conn:
        pf = ipom.predict_focussed(68)
        pf.select_prediction_data(new_format=False)
        prdct_data = pf.select_prediction_data(plan_id=13355950, new_format=True)

        pass
    # end
    print("done")


def main4():
    datasource_dict = load('datasource.json')
    url = URL.create(**datasource_dict)

    ipom = IPlanObjectModel(url)

    with ipom.connect() as conn:
        dvm = ipom.data_values_master(13355950)

        ah = dvm.area_hierarchy

        pass
    # end
    print("done")


def main3():
    datasource_dict = load('datasource.json')
    url = URL.create(**datasource_dict)

    ipom = IPlanObjectModel(url)

    with ipom.connect() as conn:
        pf = ipom.predict_focussed(68)
        train_data = pf.select_training_data(new_format=True)
        prdct_data = pf.select_prediction_data(new_format=True)

        pass
    # end
    print("done")


def main2():
    datasource_dict = load('datasource.json')
    url = URL.create(**datasource_dict)

    ipom = IPlanObjectModel(url)
    try:
        ipom.connect()

        pf = ipom.predict_focussed('IXD_NEW_IPREDICT')



        train_data = pf.select_training_data(new_format=True)

        print(train_data)
        pass
    finally:
        ipom.disconnect()


def main1():
    datasource_dict = load('datasource.json')
    url = URL.create(**datasource_dict)
    # url = "postgresql://postgres:p0stgres@10.193.20.15:5432/btdigital_ipredict_development"

    ipom = IPlanObjectModel(url)
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

        print(pf.select_training_data())

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
