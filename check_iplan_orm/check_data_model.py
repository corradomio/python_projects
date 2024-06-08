import logging.config

import pandasx as pdx
from commons import *
from iplan.om import IPlanObjectModel, DataModel
from stdlib.jsonx import load


def main():
    # datasource_dict = load('datasource.json')
    datasource_dict = load('datasource_localhost.json')
    ipom = IPlanObjectModel(datasource_dict)

    with ipom.connect():
        dmodel: DataModel = ipom.data_models().data_model(DATA_MODEL)
        print(dmodel.exists())
        print(dmodel.measure_ids())
        print(dmodel.measure_ids(with_name=True))
        print(dmodel.measures)
    pass


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
