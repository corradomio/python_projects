import random
from typing import Union

import numpy as np
import pandas as pd
import sktime.forecasting.base as skf
from pandas import DataFrame

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NoneType = type(None)

EMPTY_DICT = dict()
MAX_TS = "max_ts"
AREA = 'area'
MODEL_SCORES = ['wape', 'mape', 'r2']
INVALID_SCORES = {
    'wape': 1.0e38,
    'mape': 1.0e38,
    'r2': 1.0e38
}

TS_DATAFRAME = dict[tuple[str], DataFrame]
TS_TRAIN_PREDICT = list[tuple[tuple[str], tuple[DataFrame]]]

INPUT_TYPES = (NoneType, pd.DataFrame, np.ndarray)
TARGET_TYPES = (NoneType, pd.DataFrame, pd.Series, np.ndarray)
FH_TYPES = (NoneType, int, list, np.ndarray, pd.Index, pd.TimedeltaIndex, pd.Timedelta, skf.ForecastingHorizon)
X_EMPTY = np.zeros((0, 0))
Y_EMPTY = np.zeros(0)


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------

def dataframe_filter_outliers(df: DataFrame, col: str, outlier_std: float) -> DataFrame:
    if outlier_std <= 0:
        return df

    values = df[col].to_numpy()

    mean = np.mean(values, axis=0)
    sdev = np.std(values, axis=0)
    max_value = mean + (outlier_std * sdev)
    min_value = mean - (outlier_std * sdev)
    median = np.median(values, axis=0)

    values[(values <= min_value) | (values >= max_value)] = median

    df[col] = values
    return df
# end


# ---------------------------------------------------------------------------
# dataframe_group_split
# ---------------------------------------------------------------------------

def dataframe_split_on_groups(
    df: pd.DataFrame,
    groups: Union[None, str, list[str]],
    drop: bool = False) -> dict[tuple[str], pd.DataFrame]:
    """
    Split the dataframe based on the content area columns list

    :param df: DataFrame to split
    :param groups: list of columns to use during the split. The columns must be categorical
    :param drop: if to remove the 'groups' columns

    :return: a list [((g1,...), gdf), ...]
    """
    assert isinstance(df, (NoneType, pd.DataFrame))
    assert isinstance(groups, (NoneType, str, list))

    if df is None:
        return {}
    if isinstance(groups, str):
        groups = [groups]
    elif groups is None:
        groups = []

    if len(groups) == 0:
        return {tuple(): df}

    dfdict: dict[tuple, pd.DataFrame] = {}

    # Note: IF len(groups) == 1, Pandas return 'gname' in instead than '(gname,)'
    # The library generates a FutureWarning !!!!
    if len(groups) == 1:
        for g, gdf in df.groupby(by=groups[0]):
            dfdict[(g,)] = gdf
    else:
        for g, gdf in df.groupby(by=groups):
            dfdict[g] = gdf
    # end

    if drop:
        for g in dfdict:
            gdf = dfdict[g]
            dfdict[g] = gdf[gdf.columns.difference(groups)]
    # end

    return dfdict
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
