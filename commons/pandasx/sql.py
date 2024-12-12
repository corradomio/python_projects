import pandas as pd
import pandasx as pdx
from typing import Iterator, Optional
from pandas._libs import lib
from pandas import DataFrame
from sqlalchemy import URL, create_engine, text
from stdlib import dict_select, dict_exclude


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _jdbc_url(config: dict) -> URL:
    # jdbc:<dbms>://<host>:<port>/<database>
    surl = config["url"]
    surl = surl[5:]
    return _sqlalchemy_url(config | dict(url=surl))


def _sqlalchemy_url(config: dict) -> URL:
    # <drivername>://<host>:<port>/<database>
    url: str = config["url"]

    # <drivername>
    b = 0
    p = url.find('://')
    drivername = url[b:p]

    # <host>:<port>
    b = p+3
    p = url.find('/', b)
    host_port = url[b:p]
    s = host_port.find(':')
    if s == -1:
        host = host_port
        port = 5432
    else:
        host = host_port[0:s]
        port = int(host_port[s+1:])

    # <database>
    b = p+1
    database = url[b:]

    #   "ip": "10.193.20.15",
    #   "port": "5432",
    #   "user": "postgres",
    #   "password": "p0stgres",
    #   "db": "btdigital_ipredict_development",
    config = config | dict(
        ip=host,
        port=port,
        db=database,
        drivername=drivername
    )

    return _iplan_url(config)
# end


def _iplan_url(config: dict):
    # drivername: str,
    # username: Optional[str] = None,
    # password: Optional[str] = None,
    # host: Optional[str] = None,
    # port: Optional[int] = None,
    # database: Optional[str] = None,
    # query: Mapping[str, Union[Sequence[str], str]] = util.EMPTY_DICT,

    drivername = config.get("drivername", "postgresql")
    username = config.get("user")
    password = config.get("password")
    host = config.get("ip")
    port = config.get("port")
    database = config.get("db")
    return URL.create(drivername, username, password, host, port, database)
# end


def _dburl(config: dict) -> URL:
    str_url: Optional[str] = config["url"] if "url" in config else None
    url: URL = None
    if str_url is not None and str_url.startswith("jdbc:"):
        url = _jdbc_url(config)
    elif str_url is not None:
        url = _sqlalchemy_url(config)
    elif "ip" in config:
        url = _iplan_url(config)
    else:
        raise ValueError(f"Unvalid database configuration: {config}")
    return url
# end


# ---------------------------------------------------------------------------
# read_sql
# ---------------------------------------------------------------------------

READ_SQL_PARAMS = ["index_col", "coerce_float", "parse_dates", "columns",
                   "chunksize", "dtype_backend", "dtype"]


def read_sql(url: str, fillna=False, **kwargs) -> pd.DataFrame:
    config = dict(url=url) | _parse_url(url) | kwargs
    dburl = _dburl(config)
    kwargs = dict_select(config, READ_SQL_PARAMS)

    sql = _compose_sql(config)
    params = _parse_params(config)

    engine = create_engine(dburl)
    with engine.connect() as con:
        query = text(sql)
        df = pd.read_sql(sql=query, con=con, params=params, **kwargs)

    if fillna:
        df.fillna(pd.NA, inplace=True)

    return df
# end

# ---------------------------------------------------------------------------

def _parse_url(url: str) -> dict:
    parts = dict(params={})
    p = url.find('?')
    if p != -1:
        url = url[:p]
        qargs = url[p+1:]
    else:
        qargs = ""

    p = url.find("://")
    parts["drivername"] = url[:p]
    url = url[p+3:]
    p = url.find(":")
    parts["host"] = url[:p]
    url = url[p+1:]
    p = url.find("/")
    parts["port"] = int(url[:p])
    url = url[p+1:]
    p = url.find("/")
    if p == -1:
        parts["database"] = url
    else:
        parts["database"] = url[:p]
        parts["table"] = url[p+1:]
    # end
    if qargs.startswith("table="):
        parts["table"] = qargs[6:]
    if qargs.startswith("sql="):
        for kv in qargs.split("&"):
            k, v = kv.split("=")

            # convert lists in tuples!
            # this is necessary for 'SELECT ... FROM ... WHERE col IN :param ...'
            if k == "sql":
                parts["sql"] = v
            else:
                parts["params"][k]= v
    # end
    return parts
# end


def _parse_params(config: dict) -> dict:
    params = config["params"]
    for k in params.keys():
        params[k] = _safe_val(params[k])
    return params
# end


def _safe_val(x: str):
    if isinstance(x, list):
        return tuple(map(_safe_val, x))

    try:
        return float(x)
    except:
        pass
    try:
        return int(x)
    except:
        pass
    if x.startswith("'") or x.startswith('"'):
        return x[1:-1]
    else:
        return x
# end


def _compose_sql(config: dict) -> str:
    if "table" in config:
        table = config["table"]
        return f"SELECT * FROM {table}"
    if "sql" in config:
        sql = config["sql"]
        return sql
    else:
        raise ValueError(f"Unsupported sql configuration: {config}")


# def _read_sql(
#     sql,
#     con,
#     index_col=None,
#     coerce_float=True,
#     params=None,
#     parse_dates=None,
#     columns=None,
#     chunksize=None,
#     dtype_backend=lib.no_default,
#     dtype=None,
# ) -> DataFrame | Iterator[DataFrame]:
#     """
#     It extends 'pandas.read_sql' forcing all 'Not a Number' values to be 'pd.NA'
#
#     Note: pd.NA is an experimental value. Why it is used this!
#     """
#     df = pd.read_sql(
#         sql, con,
#         index_col=index_col,
#         coerce_float=coerce_float,
#         params=params,
#         parse_dates=parse_dates,
#         columns=columns,
#         chunksize=chunksize,
#         dtype_backend=dtype_backend,
#         dtype=dtype
#     )
#     df.fillna(pd.NA, inplace=True)
#     return df
# # end


# def _read_sql_query(
#     sql,
#     con,
#     index_col=None,
#     coerce_float=True,
#     params=None,
#     parse_dates=None,
#     chunksize=None,
#     dtype=None,
#     dtype_backend=lib.no_default,
# ) -> DataFrame | Iterator[DataFrame]:
#     """
#     It extends 'pandas.read_sql_query' forcing all 'Not a Number' values to be 'pd.NA'
#
#     Note: pd.NA is an experimental value. Why it is used this!
#     """
#
#     df = pd.read_sql_query(
#         sql, con,
#         index_col=index_col,
#         coerce_float=coerce_float,
#         params=params,
#         parse_dates=parse_dates,
#         chunksize=chunksize,
#         dtype=dtype,
#         dtype_backend=dtype_backend,
#     )
#     df.fillna(pd.NA, inplace=True)
#     return df
# # end


# ---------------------------------------------------------------------------
# write_sql
# ---------------------------------------------------------------------------

def write_sql(df: pd.DataFrame, url: str, **kwargs):
    config = dict(url=url) | kwargs

    _clean_table(df, config)
    _write_table(df, config)
# end

# ---------------------------------------------------------------------------

TO_SQL_PARAMS = ["schema", "if_exists", "index", "index_label", "chunksize",
                 "dtype", "method"]


def _where_clause(groups: list[str]):
    where = ""
    for g in groups:
        if len(where) == 0:
            where = f" WHERE {g}=:{g}"
        else:
            where += f" AND {g}=:{g}"
    return where
# end


def _where_params(groups: list[str], g: tuple) -> dict:
    n = len(groups)
    params = {}
    for i in range(n):
        params[groups[i]] = g[i]
    return params
# end


def _clean_table(df: pd.DataFrame, config: dict):
    dburl = _dburl(config)
    groups: list[str] = config.get("groups", [])
    table: str = config["table"]
    where = _where_clause(groups)
    sql = text(f"DELETE FROM {table}{where}")
    engine = create_engine(dburl)

    if len(groups) == 0:
        try:
            with engine.connect() as con:
                con.execute(sql)
                con.commit()
        finally:
            engine.dispose()
        return
    # end

    g_list: list[tuple] = pdx.groups_list(df, groups=groups)
    try:
        for g in g_list:
            params = _where_params(groups, g)
            with engine.connect() as con:
                con.execute(sql, parameters=params)
                con.commit()
    finally:
        engine.dispose()
# end


def _write_table(df: pd.DataFrame, config: dict):
    dburl = _dburl(config)
    table: str = config["table"]
    kwargs: dict = dict_select(config, TO_SQL_PARAMS)

    engine = create_engine(dburl)
    try:
        with engine.connect() as con:
            df.to_sql(table, con=con, if_exists="append", index=False, **kwargs)
    finally:
        engine.dispose()
    return
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
