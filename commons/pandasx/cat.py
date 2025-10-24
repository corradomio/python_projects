from typing import Union

import pandas as pd
from pandas import CategoricalDtype

__all__ = ['find_categorical_columns', 'unique_values']


# ---------------------------------------------------------------------------
# concat
# ---------------------------------------------------------------------------

def find_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Select the categorical columns
    """
    columns = []
    for col in df.columns:
        dtype = df[col].dtype
        if isinstance(dtype, CategoricalDtype):
            columns.append(col)
        elif dtype in [str]:
            columns.append(dtype)
        else:
            pass
    return columns


def unique_values(df: pd.DataFrame, columns: Union[None, str, list[str]] = None) -> dict[str, list[str]]:
    """
    For each categorical column return the list of unique values
    """
    uvalues = {}
    if columns is None:
        columns = find_categorical_columns(df)

    for col in columns:
        uvalues[col] = list(df[col].unique())
    return uvalues


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
