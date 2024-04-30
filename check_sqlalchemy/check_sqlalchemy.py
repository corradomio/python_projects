import sqlalchemy as sqla
from sqlalchemy import Column, Integer, Float, Text, Table, BigInteger
from sqlalchemy import MetaData
from sqlalchemy import insert
from sqlalchemy.orm import declarative_base

TablesBase = declarative_base()


class MODELS_SCORES(TablesBase):
    __tablename__ = "ip_train_models"

    index: int = Column('index', BigInteger(), primary_key=True, nullable=False, autoincrement=True)
    model: str = Column('model', Text(), nullable=False)

    order: int = Column('order', Integer(), nullable=False)
    mape: float = Column('mape', Float(), nullable=False)
    wape: float = Column('wape', Float(), nullable=False)
    r2: float = Column('r2', Float(), nullable=False)
    area: str = Column('area', Text(), nullable=True)
# end


def drop_table(table_name, engine):
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table: Table = metadata.tables[table_name]
    if table is not None:
        # Base.metadata.drop_all(engine, [table], checkfirst=True)
        table.drop(bind=engine)


def create_scores_table(table_name, engine):
    schema = MODELS_SCORES
    meta = MetaData()
    table = Table(table_name, meta)

    for col in schema.__table__.columns:
        print(col)
        table.append_column(Column(col.name, col.type, nullable=col.nullable, primary_key=col.primary_key, autoincrement=col.autoincrement))

    table.create(bind=engine)
    pass



def create_predict_table(table_name, schema, engine):
    pass


def get_table(table_name: str, engine: sqla.engine.Engine):
    metadata = sqla.MetaData()
    metadata.reflect(bind=engine)
    return metadata.tables[table_name]
# end


def main():
    url = "mysql+mysqlconnector://root@localhost:3306/ipredict4"
    engine = sqla.create_engine(url, echo=False)
    inspector: sqla.engine.Inspector = sqla.inspect(engine)
    # if inspector.has_table('algo_scores_1'):
    #     drop_table('algo_scores_1', engine)
    if not inspector.has_table('algo_scores_1'):
        create_scores_table('algo_scores_1', engine)

    t = get_table('algo_scores_1', engine)

    with engine.connect() as c:
        stmt = insert(t).values(
            index=0,
            model="a model",
            order=101,
            mape=11,
            wape=22,
            r2=33,
            area="cicciopasticcio"
        )
        print(stmt)
        c.execute(
                stmt
                # model="a model",
                # order=101,
                # mape=11,
                # wape=22,
                # r2=33,
                # area="cicciopasticcio"
        )
        c.commit()
    pass
    pass


if __name__ == "__main__":
    main()
