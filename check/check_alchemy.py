import sqlalchemy
import psycopg2
from sqlalchemy import inspect
from sqlalchemy.engine import create_engine

# {'DATE', 'TIMESTAMP', 'DOUBLE_PRECISION', 'TEXT', 'NUMERIC', 'BIGINT', 'INTEGER', 'VARCHAR', 'VARCHAR(256)'}

print(sqlalchemy.__version__)
print(psycopg2.__version__)

url = "postgresql://postgres:p0stgres@10.193.20.15:5432/adda"
COLUMN_TYPES = set()

engine = create_engine(url)
engine.connect()
# print (engine.table_names())

try:
    inspector = inspect(engine)
    schemas = inspector.get_schema_names()
    for schema in schemas:
        print(f"schema: {schema}")
        for table_name in inspector.get_table_names(schema=schema):
            print(f"  {table_name}")
            for column in inspector.get_columns(table_name, schema=schema):
                print(f"    {column}")
                COLUMN_TYPES.add(str(column['type']))
except Exception as e:
    pass
finally:
    engine.dispose()

print(COLUMN_TYPES)
