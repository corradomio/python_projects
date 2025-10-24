from sqlalchemy import MetaData, Table, Column, inspect
from sqlalchemy.engine import Engine, Inspector


# ---------------------------------------------------------------------------
# Simple Table operations
# ---------------------------------------------------------------------------

def get_table(table_name: str, engine: Engine):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    return metadata.tables[table_name]
# end


def exists_table(table_name: str, engine: Engine):
    inspector: Inspector = inspect(engine)
    return inspector.has_table(table_name)
# end


def drop_table(table_name: str, engine: Engine):
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table: Table = metadata.tables[table_name]
    if table is not None:
        table.drop(bind=engine)
# end


def create_table(table_name: str, schema, engine: Engine):
    meta: MetaData = MetaData()
    table = Table(table_name, meta)

    # for col in schema.__table__.columns:
    for col in schema.__columns__:
        table.append_column(Column(col.name, col.type, nullable=col.nullable))

    table.create(bind=engine)
    return table
# end
