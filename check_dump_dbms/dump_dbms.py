import logging.config
import os

from sqlalchemy import create_engine, inspect
import pandas as pd
from joblib import Parallel, delayed


def dump_table(conn, table_name, dbname):
    try:
        # df = pd.read_sql_table(table_name, conn, schema='public')
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df.to_csv(f"{dbname}/{table_name}.csv", index=False)
    except Exception as e:
        print(f"ERROR {table_name}:", e)


SKIP_TABLES = [
    # "tb_idata_values_detail"
]

def main():
    # dbname = "btdigital_ipredict_development"
    dbname = "adda"
    engine = create_engine(f"postgresql://postgres:p0stgres@10.193.20.15:5432/{dbname}", echo=False)
    conn = engine.connect()
    os.makedirs(dbname, exist_ok=True)

    inspector = inspect(engine)
    schemas = inspector.get_schema_names()
    for schema in schemas:
        if schema == 'public':
            print("schema: %s" % schema)
            for table_name in inspector.get_table_names(schema=schema):
                if table_name in SKIP_TABLES:
                    print("... skipped: %s" % table_name)
                    continue
                else:
                    print("... table: %s" % table_name)

                dump_table(conn, table_name, dbname)
                # for column in inspector.get_columns(table_name, schema=schema):
                #     print("... ... column: %s" % column)



if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
