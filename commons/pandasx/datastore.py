from typing import Union

import pandas as pd
from path import Path as path

import pandasx as pdx
from stdlib import dict_select, dict_exclude
from stdlib.is_instance import is_instance
from .datasource import _compose_url, _filepath


# ---------------------------------------------------------------------------
# save_to
# ---------------------------------------------------------------------------

def write_to(df: pd.DataFrame, config: Union[dict, str]):
    assert is_instance(config, Union[dict, str]), "Invalid config type: supported str, dict"

    #
    # <url>: str -> { "url": <url> }
    #
    if isinstance(config, str):
        config = dict(url=config)

    #
    # {
    #    "datastore": {...}
    # }
    if isinstance(config, dict) and "datastore" in config:
        config = config["datastore"]

    #
    # database url specified in "IPlan" format:
    #   "ip", "port", "db"
    #   "drivername" (optional)
    #
    if "ip" in config:
        config["url"] = _compose_url(config)
        config = dict_exclude(config, ["drivername", "ip", "port", "db"])

    #
    # save the data
    #
    url = config["url"]
    assert isinstance(url, str), f"Invalid url: {url}"

    if url.startswith("file://"):
        return _write_to_file(df, config)
    if url.startswith("inline:") or url.startswith("memory:"):
        return _write_to_inline(df, config)
    if url.find("://") != -1:
        # jdbc:posgresql://<host>:<port>/<database>
        # posgresql://<host>:<port>/<database>
        return _write_to_database(df, config)
    else:
        raise ValueError(f"Unsupported url '{url}'")
# end


# ---------------------------------------------------------------------------
# _write_to_file
# ---------------------------------------------------------------------------

TO_FILE_PARAMS = ["sep", "na_rep", "float_format", "columns", "header", "index", "index_label", "mode", "encoding",
                  "compression", "quoting", "quotechar", "lineterminator", "chunksize", "date_format", "doublequote",
                  "escapechar", "decimal", "errors", "storage_options"]

def _write_to_file(df: pd.DataFrame, config: dict) -> dict:
    absolute_path = path(_filepath(config["url"])).absolute()
    kwargs = dict_select(config, TO_FILE_PARAMS)

    pdx.write_data(df, absolute_path, **kwargs)
    return {
        "filepath": absolute_path,
        "records": len(df)
    }
# end


# ---------------------------------------------------------------------------
# _write_to_database
# ---------------------------------------------------------------------------
# Before to write in the database, it is necessary to delete the old data
# We suppose that the table contains on data organized by 'groups', NOT by date
#

def _write_to_database(df: pd.DataFrame, config: dict) -> dict:
    url = config["url"]
    kwargs = dict_exclude(config, ["url"])
    pdx.write_data(df, url, **kwargs)

    return {
        "table": config["table"],
        "records": len(df)
    }
# end


# ---------------------------------------------------------------------------
# _write_to_inline
# ---------------------------------------------------------------------------

TO_JSON_PARAMS = [
    "date_format", "double_precision", "force_ascii", "date_unit", "default_handler",
    "lines", "compression", "index", "indent", "storage_options", "mode",
    "na_values", "datetime"     # extended parameters
]

def _write_to_inline(df: pd.DataFrame, config: dict) -> dict:
    orient = config['format']
    kwargs = dict_select(config, TO_JSON_PARAMS)
    jdata = pdx.write_data(df, "inline:", orient=orient, **kwargs)
    return {
        "format": orient,
        "data": jdata
    }
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------




