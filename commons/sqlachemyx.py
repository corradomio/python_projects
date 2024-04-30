from sqlalchemy import *
import sqlalchemy as sqla


def get_table(table_name: str, engine: sqla.engine.Engine):
    metadata = sqla.MetaData()
    metadata.reflect(bind=engine)
    return metadata.tables[table_name]
# end


def exists_table(table_name: str, engine: sqla.engine.Engine):
    inspector: sqla.engine.Inspector = sqla.inspect(engine)
    return inspector.has_table(table_name)
# end


def drop_table(table_name: str, engine: sqla.engine.Engine):
    metadata = sqla.MetaData()
    metadata.reflect(bind=engine)
    table: sqla.Table = metadata.tables[table_name]
    if table is not None:
        table.drop(bind=engine)
# end


def create_table(table_name: str, schema, engine: sqla.engine.Engine):
    meta: sqla.MetaData = sqla.MetaData()
    table = sqla.Table(table_name, meta)

    # for col in schema.__table__.columns:
    for col in schema.__columns__:
        table.append_column(sqla.Column(col.name, col.type, nullable=col.nullable))

    table.create(bind=engine)
    return table
# end
