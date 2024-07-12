#
# Note: now, 'jsonx' is able to serialize in JSON format
#       several 'almost-standard' data types:
#
#       Python data/datetime
#       Pandas Series, DataFrame
#       Numpy arrays
#
#
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Literal, Optional
from typing import Union

import pandas as pd


def to_json(
        data: Union[pd.Series, pd.DataFrame],
        path: Optional[str] = None,
        orient: Optional[Literal["split", "records", "index", "table", "columns", "values"]] = None,
        **kwargs) -> dict:

    assert isinstance(data, (pd.DataFrame, pd.Series))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if path is None:
        json_file = f"tmp-{timestamp}.json"
    else:
        json_file = path

    data.to_json(json_file, orient=orient, **kwargs)

    if path is None:
        with open(json_file, mode='r') as t:
            jdata = json.load(t)
        os.remove(json_file)
    else:
        jdata = None

    return jdata
# end
