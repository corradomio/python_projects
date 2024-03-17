from typing import Union
from pandas import DataFrame
from pandas.errors import PerformanceWarning
import warnings

from .base import as_list, safe_sorted


# ---------------------------------------------------------------------------
# onehot_encode
# ---------------------------------------------------------------------------

def onehot_encode(df: DataFrame, columns: Union[str, list[str]]) -> DataFrame:
    assert isinstance(df, DataFrame)
    columns = as_list(columns, 'columns')

    warnings.simplefilter(action='ignore', category=PerformanceWarning)

    for col in columns:
        # check if the column contains ONLY 2 values
        uv = safe_sorted(df[col].unique())
        nv = len(uv)

        if nv <= 2:
            vmap = {uv[i]: i for i in range(nv)}
            df[col] = df[col].replace(vmap)
            continue

        # prepare the columns
        for v in uv:
            ohcol = f"{col}_{v}"
            df[ohcol] = 0

        # fill the columns
        for v in uv:
            ohcol = f"{col}_{v}"
            df.loc[df[col] == v, ohcol] = 1

        # remove column & clone to reduce fragmentation
        # df = df[df.columns.difference([col])].copy()
        df.drop([col], axis=1, inplace=True)
        pass
    return df
# end
