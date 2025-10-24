from sqlalchemy import MetaData, Table, ForeignKeyConstraint, PrimaryKeyConstraint
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from networkx import MultiDiGraph

TablesBase = declarative_base()


def main():
    url = "postgresql://postgres:p0stgres@10.193.20.15:5432/btdigital_ipredict_development"
    engine = create_engine(url, echo=False)
    metadata = MetaData()
    metadata.reflect(engine)

    dg = MultiDiGraph()

    for t_name in metadata.tables:
        t: Table = metadata.tables[t_name]
        print(t)
        for fk in t.foreign_key_constraints:
            r: Table = fk.referred_table
            r_pk: list[str] = [c.name for c in r.primary_key.c]
            print(f"... {fk.column_keys[0]} -> {fk.referred_table}:{r_pk[0]}")


    pass



if __name__ == "__main__":
    main()