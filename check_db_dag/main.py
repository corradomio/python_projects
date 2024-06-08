import logging.config
import networkx as nx
import sqlalchemy as sa
import stdlib
from stdlib.jsonx import load
from dbdag import DatabaseDAG


def main():
    log = logging.getLogger("main")
    log.info(f"nx: {nx.__version__}")
    log.info(f"sqlalchemy: {sa.__version__}")
    log.info(f"stdlib: {stdlib.__version__}")

    # datasource_dict = load('datasource.json')
    datasource_dict = load('datasource_localhost.json')
    ddag = DatabaseDAG(datasource_dict)

    with ddag.connect():

        # table: Table = ddag.iDataMaster
        # query = select(table)
        # log.debug(query)
        # query = query.where(table.c.area_id_fk == 1)
        # log.debug(query)
        # query = query.where(table.c.skill_id_fk == 1)
        # log.debug(query)
        ddag.scan_all()
        # ddag.scan(ddag.iDataMaster)


if __name__ == "__main__":
    logging.config.fileConfig("logging_config.ini")
    logging.getLogger("root").info("Logging initialized")
    main()
