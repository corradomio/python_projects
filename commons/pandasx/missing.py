#
# strategies to replace missing values
#
#   1) mean | median
#   2) interpolation if the there are an maximum
#
#   np.nan
#   None
#   NaN         (float)         float('nan'), float('NaN'), float('NAN')
#   pd.NA       (integer)       <NA>
#   pd.NaT      (timestamp)     NaT
#
#   df.fillna(value)
#   df.fillna(method="pad")
#   df.fillna(method="pad", limit=1)
#       pad/ffill       forward
#       bfill/backfill  backward
#
#   df.interpolate(method="barycentric", order:int, limit:int, limit_direction, limit_area)
#       method: barycentric pchip akima
#
#   df.replace(val, val)
#       value, value
#       [v1,...], [r1,...]
#       {v1: r1, ...}
#       {"c1": v1, ...}, r1
#
#   s.isna()                    -> series[bool]
#   s.notna()                   -> series[bool]
#   df.isna()                   -> dataframe[bool]
#   df.notna()                  -> dataframe[bool]
#
#   s/df.isna().values.any()    -> bool
#   s.isnull().sum()            -> int
#   df.isnull().sum().sum()     -> int
#   df.isnull().sum()           -> series[column: count]
#
#   pd.isnull(s/df) == pd.isna(s/df)
#   pd.isna(s/df)   == df.isna(s/df)
#
#   df.fillna(method="bfill)    == df.bfill()
#   df.fillna(method="ffill)    == df.ffill()
#
#   df.fillna(df.mean()) / df.fill(df.median())
#
from typing import Union
import pandas as pd
from .base import as_list


def nan_replace(df: pd.DataFrame,
                dropcol=.0,
                dropna=False,
                fillna=None,
                fill_limit=None,
                interpolate=None, order=None,
                interpolate_limit=None,
                limit_direction=None,
                limit_area=None,
                columns=None,
                ignore=None) -> pd.DataFrame:
    """
    Apply some strategies to remove missing values:

        0) it is used 'dropna' if enabled
        1) it is used 'interpolate' if specified
        2) it is used 'fillna' if specified

    If 'fillna' method is 'mean' or 'median', it is used the mean or median computed before
    to apply the interpolation
    
    :param df:
    :param dropcol: drop a column if it contains more of 'dropcol*100'% of nan
    :param fillna: fillna method (value, "pad", "ffill", "bfill", "backfill", "mean", "median")
    :param fill_limit: fillna limit
    :param interpolate: interpolate method
            "linear", "quadratic", "cubic"
            "time", "values", "quadratic", "pchip", "akima", "barycentric"
            "spline", "polynomial"
            and available 'scipy.interpolate' methods
    :param order: "spline", "polynomial" interpolation order
    :param interpolate_limit: maximum length of consecutive missing values
    :param limit_direction: interpolate limit direction ("forward", "backward", "both")
    :param limit_area: interpolate limit area ("inside", "outside")
    :param columns: columns to analyze (alternative to 'ignore')
    :param ignore: list of columns to exclude (alternative to 'columns')
    :return: 
    """
    ignore = as_list(ignore, 'ignore')
    columns = as_list(columns, 'columns')

    df = df.copy()

    # drop the columns if necessary
    dropcols = _invalid_columns(df, quota=dropcol, columns=columns, ignore=ignore)
    if len(dropcols) > 0:
        df.drop(dropcols, inplace=True)

    # if fillna is 'mean' or 'median', compute the 'real' value based on the
    # current dataframe's content
    if fillna == "mean":
        fill_value = df.mean()
    elif fillna == "median":
        fill_value = df.median()
    elif not isinstance(fillna, str):
        fill_value = fillna
    else:
        fill_value = None

    # apply dropna if enables
    if dropna:
        df.dropna(how='any', axis=0, inplace=True)

    # no interpolate/fillna method defined
    if interpolate is None and fillna is None:
        assert df.isnull().sum().sum() == 0, "The dataframe contains NaN/NaT/None values"
        return df

    # no nan present
    if df.isnull().sum().sum() == 0:
        return df

    # select the sub-dataframe
    if len(columns) > 0:
        dfnan = df[columns].copy()
    elif len(ignore) > 0:
        dfnan = df[df.columns.difference(ignore)].copy()
    else:
        dfnan = df

    # try to apply interpolate
    if interpolate is not None:
        dfnan.interpolate(method=interpolate,
                          limit=interpolate_limit,
                          limit_direction=limit_direction,
                          limit_area=limit_area,
                          order=order,
                          axis=1,
                          inplace=True)

    # try to apply fillna
    if fillna in ["mean", "median"] or not isinstance(fillna, str):
        dfnan.fillna(value=fill_value, limit=fill_limit, inplace=True)
    elif fillna in ["backfill", "bfill", "ffill", "pad"]:
        dfnan.fillna(method=fillna, limit=fill_limit, inplace=True)
    else:
        raise ValueError(f"Unsupported fillna method '{fillna}'")

    # update the original dataframe, if necessary
    if len(df.columns) > len(dfnan.columns):
        df[dfnan.columns] = dfnan

    # in 'theory' ALL nan must be resolved
    assert df.isnull().sum().sum() == 0, "The dataframe contains NaN/NaT/None values"

    return df
# end77


def _invalid_columns(df: pd.DataFrame, quota: Union[float, int], columns: list[str], ignore: list[str]):
    # if quota = 0, skip the check
    if quota == 0:
        return []

    # select the columns to analyze
    if len(ignore) > 0:
        columns = list(df.columns.difference(ignore))
    elif len(columns) == 0:
        columns = list(df.columns)
    pass

    # compute the number of nans that makes the column 'invalid'
    n = len(df)
    q = quota if quota >= 1 else int(quota*n)

    # analyze the columns
    icols = []
    for col in columns:
        nnulls = df[col].isnull().sum()
        if nnulls >= q:
            icols.append(col)
    # end
    return icols
# end