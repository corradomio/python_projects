import os
import logging.config
from sqlalchemy import create_engine, inspect
import pandas as pd

def dump_table(conn, table_name, dbname):
    try:
        # df = pd.read_sql_table(table_name, conn, schema='public')
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(f"{dbname}/{table_name}.csv", index=False)
    except Exception as e:
        print(f"ERROR {table_name}:", e)




def main():
    # dbname = "btdigital_ipredict_development"
    dbname = "etsalat_fleet"
    os.makedirs(dbname, exist_ok=True)
    engine = create_engine(f"postgresql://postgres:p0stgres@10.193.20.15:5432/{dbname}", echo=False)
    conn = engine.connect()
    schema = 'public'

    for table_name in ['tb_idata_model_detail', 'tb_services_invoke_scheduler']:
        print("... table: %s" % table_name)
        dump_table(conn, table_name, dbname)
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
