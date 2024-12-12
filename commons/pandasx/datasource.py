import os
from typing import Union, Optional

from datetime import datetime
import pandas as pd
from sqlalchemy import URL, create_engine, text

import pandasx as pdx
from stdlib.is_instance import is_instance
from stdlib.dictx import dict_exclude
import json
from path import Path as path


# ---------------------------------------------------------------------------
# read_from
# ---------------------------------------------------------------------------

def read_from(config: Union[dict, str]) -> pd.DataFrame:
    assert is_instance(config, Union[dict, str]), "Invalid config type: supported str, dict"

    #
    # <url>: str -> { "url": <url> }
    #
    if isinstance(config, str):
        config = dict(url=config)

    #
    # {
    #    "datasource": {...}
    # }
    if isinstance(config, dict) and "datasource" in config:
        config = config["datasource"]

    #
    # database url specified in "IPlan" format:
    #   "ip", "port", "db"
    #   "drivername" (optional)
    #
    if "ip" in config:
        config["url"] = _compose_url(config)
        config = dict_exclude(config, ["drivername", "ip", "port", "db"])

    #
    # read the data
    #
    url = config["url"]
    assert isinstance(url, str), f"Invalid url: {url}"

    if url.startswith("file://"):
        return _read_from_file(config)
    if url.startswith("inline:") or url.startswith("memory:"):
        return _read_from_inline(config)
    if url.find("://") != -1:
        # jdbc:posgresql://<host>:<port>/<database>
        # posgresql://<host>:<port>/<database>
        return _read_from_database(config)
    else:
        raise ValueError(f"Unsupported url '{url}'")
# end


def _compose_url(config: dict) -> str:
    drivername = config.get("drivername", "postgresql")
    ip = config["ip"]
    port = config["port"]
    db = config["db"]
    return f"{drivername}://{ip}:{port}/{db}"
# end


# ---------------------------------------------------------------------------
# _read_from_file
# ---------------------------------------------------------------------------

def _filepath(url: str) -> str:
    if url.startswith("file:///"):
        return url[8:]
    if url.startswith("file://"):
        return url[7:]
    raise ValueError(f"Unsupported url {url}")


def _read_from_file(config: dict)-> pd.DataFrame:
    absolute_path = path(_filepath(config["url"])).absolute()
    params = dict_exclude(config, ["url"])

    df = pdx.read_data(absolute_path, **params)
    return df


# ---------------------------------------------------------------------------
# _read_from_inline
# ---------------------------------------------------------------------------

SUPPORTED_ORIENTS = {'split', 'records', 'index', 'columns', 'values', 'table'}

#
#   {
#       "columns": [...]
#       "index": [...]
#       "data": [[...], ... ]
#   }
#
def _na_values(data_dict, na_values) -> dict:
    if len(na_values) == 0:
        return data_dict

    data = data_dict["data"]

    n = len(data)
    m = len(data[0]) if n > 0 else 0
    for i in range(n):
        for j in range(m):
            if data[i][j] in na_values:
                data[i][j] = None
    # end

    data_dict["data"] = data

    return data_dict
# end


def _to_df(data: dict, orient: str) -> pd.DataFrame:
    # note: it is necessary to save the JSON in a file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    json_file = f"tmp-{timestamp}.json"

    with open(json_file, mode="w") as fp:
        json.dump(data, fp)

    with open(json_file) as fp:
        df = pd.read_json(fp, orient=orient)

    os.remove(json_file)
    return df
# end


def _guess_orient(data: dict):
    if "columns" in data and "data" in data and "index" in data:
        return "split"
    else:
        return "no format"
# end


#
#   {
#       "url": "inline:///<format>"
#       "format": <format>
#       "data": {
#           "columns": [...]
#           "index": [...]
#           "data": [
#               [...],
#               ...
#           ]
#       ...
#   }
#
def _read_from_inline(config: dict)-> pd.DataFrame:
    assert isinstance(config, dict)
    assert "url" in config, "Missing 'url' parameter"
    assert "data" in config, "Missing 'data' parameter"

    url: str = config["url"]
    data = config["data"]
    na_values = config.get("na_values", [])

    # retrieve the JSON data 'format:
    # 1) from "url"
    #   inline:///<format>
    #   memory:///<format>
    # 2) from "format" parameter
    #
    if "format" in config:
        orient = config["format"]
    else:
        p = url.rfind('/')
        orient = url[p+1:]

    # guess the format based in the tags in 'config["data"]
    if orient in [None, ""]:
        orient = _guess_orient(data)

    # Supported formats:
    #   https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
    #
    #   ['split', 'records', 'index', 'columns', 'values', 'table']
    #
    #
    if not orient in SUPPORTED_ORIENTS:
        raise ValueError(f"Unknown data format '{orient}'")

    # parse null values
    data = _na_values(data, na_values)

    # convert {...} into a dataframe
    df = _to_df(data, orient)

    # apply cleanup
    kwargs = dict_exclude(config, ["url", "format", "data"])
    df = pdx.cleanup_data(df, **kwargs)

    return df
# end


# ---------------------------------------------------------------------------
# _read_from_database
# ---------------------------------------------------------------------------
# Java/JDBC
#
#   "url": "jdbc:posgrsql://<host>:<port>/<database>"
#   "user": <userbname>,
#   "password": <password>,
#
# Python/SQLAlchemy
#
#   "url": "postgresql://<host>:<port>/<database>"
#   "user": <userbname>,
#   "password": <password>,
#
# Java/IPlan
#   "ip": "10.193.20.15",
#   "port": "5432",
#   "user": "postgres",
#   "password": "p0stgres",
#   "db": "btdigital_ipredict_development",
#
#
#   "table": <table_name>
#   "sql": "SELECT ..."
#   "params": { ... }

def _from_jdbc_url(config: dict) -> URL:
    # jdbc:<dbms>://<host>:<port>/<database>
    surl = config["url"]
    surl = surl[5:]
    return _from_sqlalchemy_url(config | dict(url=surl))


def _from_sqlalchemy_url(config: dict) -> URL:
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

    return _from_iplan_url(config)


def _from_iplan_url(config: dict):
    #         drivername: str,
    #         username: Optional[str] = None,
    #         password: Optional[str] = None,
    #         host: Optional[str] = None,
    #         port: Optional[int] = None,
    #         database: Optional[str] = None,
    #         query: Mapping[str, Union[Sequence[str], str]] = util.EMPTY_DICT,

    drivername = config.get("drivername", "postgresql")
    username = config.get("user")
    password = config.get("password")
    host = config.get("ip")
    port = config.get("port")
    database = config.get("db")
    return URL.create(drivername, username, password, host, port, database)


def _get_dbms_url(config: dict) -> URL:
    str_url: Optional[str] = config["url"] if "url" in config else None
    url: URL = None
    if str_url is not None and str_url.startswith("jdbc:"):
        url = _from_jdbc_url(config)
    elif str_url is not None:
        url = _from_sqlalchemy_url(config)
    elif "ip" in config:
        url = _from_iplan_url(config)
    else:
        raise ValueError(f"Unvalid database configuration: {config}")

    return url


def _read_from_database(config: dict)-> pd.DataFrame:
    dbms_url = str(_get_dbms_url(config))
    kwargs = dict_exclude(config, ["url", "user", "password"])
    return pdx.read_data(dbms_url, **kwargs)
# end


def _read_from_database_old(config: dict)-> pd.DataFrame:
    dburl = _get_dbms_url(config)
    sql: str = ""
    params: dict = config.get("params",{})

    if "table" in config:
        table = config["table"]
        sql = f"SELECT * FROM {table}"
    elif "sql" in config:
        sql = config["sql"]
    else:
        ValueError("Invalid datasource: missing table or sql statemenet")

    engine = create_engine(dburl)
    try:
        with engine.connect() as con:
            df = pd.read_sql(text(sql), con, params=params)
    finally:
        engine.dispose()

    return df
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
