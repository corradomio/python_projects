import logging
from typing import Optional, Union
import json

import pandas as pd
import sqlalchemy.engine as sae


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def is_filesystem(datasource: str) -> bool:
    # file://....
    # <disk>:....
    if datasource.startswith("file://") or len(datasource) > 2 and datasource[1] == ':':
        return True
    elif "://" in datasource:
        return False
    else:
        raise ValueError(f"Unsupported datasource '{datasource}'")


def dict_select(d: dict, keys: list[str]) -> dict:
    s = {}
    for k in keys:
        if k in d:
            s[k] = d[k]
    return s
# end


# ---------------------------------------------------------------------------
# DatasourceLoader
# ---------------------------------------------------------------------------
# Note:
#   because the data is loaded from a file or a database table, it is present
#   the 'datetime' column.
#
#   In a file, the column is string, BUT in a database's table the column can be
#   of type datetime (or other equivalent type) OR a string


def _normalize_datasource(datasource) -> dict:
    if isinstance(datasource, str):
        return dict(url=datasource)
    assert isinstance(datasource, dict)
    assert 'url' in datasource

    url: Union[str, dict] = datasource['url']
    if isinstance(url, str):
        return datasource

    table = url['table']
    url_config = dict_select(url, ['drivername', 'username', 'password', 'host', 'port', 'database'])
    url: sae.URL = sae.URL.create(**url_config)
    surl: str = f"{str(url)}/{table}"

    datasource = {} | datasource
    datasource['url'] = surl
    return datasource
# end


class DatasourceLoader:

    # -----------------------------------------------------------------------
    # Factory method
    # -----------------------------------------------------------------------
    #   datasource  = <url>
    #               | { 'url': <url>, 'params': { 'name': value, ... } }
    #               | { 'url': { ... }, 'params': { 'name': value, ... } }
    #

    @staticmethod
    def from_file(config_file: str) -> "DatasourceLoader":
        with open(config_file) as fp:
            config = json.load(fp)
            return DatasourceLoader.from_config(config)
    # end

    @staticmethod
    def from_config(config: dict) -> "DatasourceLoader":
        datasource = _normalize_datasource(config)

        url = datasource['url']
        if is_filesystem(url):
            return FilesystemDatasourceLoader(datasource)
        else:
            return SQLAlchemyDatasourceLoader(datasource)
    # end

    # deprecated: use 'from_config(...)'
    @staticmethod
    def from_datasource(config: dict) -> "DatasourceLoader":
        return DatasourceLoader.from_file(config)
    # end

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    """
    Load the data from different urls:

        file:///__absolute_path__
        __absolute_path__

        <dbms-protocol>://<host>:<port>.<database>/<table>
    """

    def __init__(self):
        self.datasource: Optional[dict] = None
        self._log = logging.getLogger("ipredict.DatasourceLoader")

    def load(self) -> pd.DataFrame:
        dsdict = self.datasource

        url = dsdict["url"]
        self._log.info(f"load data from '{url}'")

        df = self._load(dsdict)

        # automatically remove nans ONLY in columns NOT 'target'
        # subset = list(df.columns.difference([target]))
        # df = df.dropna(subset=subset)

        self._log.info(f"... {df.shape}: {list(df.columns)}")
        return df

    def _load(self, dsdict: dict) -> pd.DataFrame:
        ...
# end


class FilesystemDatasourceLoader(DatasourceLoader):
    def __init__(self, datasource: Optional[dict]):
        super().__init__()
        self.datasource = datasource

    def _load(self, dsdict: dict) -> pd.DataFrame:
        # file:///__file_path__
        # file://__file_path__
        # __file_path__
        file = dsdict["url"]
        # file = self.datasource
        if file.startswith("file:///"):
            file = file[8:]
        elif file.startswith("file://"):
            file = file[7:]
        df = pd.read_csv(file)
        return df
# end


class SQLAlchemyDatasourceLoader(DatasourceLoader):
    def __init__(self, datasource: Optional[dict]):
        super().__init__()
        self.datasource = datasource

    def _load(self, dsdict: dict) -> pd.DataFrame:
        # dbms://host:port/database/table
        # dbms://host:port?sql=query
        url = dsdict["url"]
        params = {} if 'params' not in dsdict else dsdict['params']

        # datasource = self.datasource
        if '?sql=' in url:
            # dbms://host:port/database?sql=query
            pos = url.index('?sql=')
            dburl = url[0: pos]
            sql = url[pos+5:]
        elif '?table=' in url:
            # dbms://host:port/database?table=table_name
            pos = url.index('?table=')
            dburl = url[0: pos]
            table = url[pos+7:]
            sql = f"SELECT * FROM {table}"
        else:
            # dbms://host:port/database/table
            pos = url.rfind('/')
            dburl = url[0: pos]
            table = url[pos+1:]
            sql = f"SELECT * FROM {table}"

        from sqlalchemy import create_engine
        engine = create_engine(dburl)
        with engine.connect() as con:
            df = pd.read_sql(sql=sql, con=con, params=params)
        return df
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
