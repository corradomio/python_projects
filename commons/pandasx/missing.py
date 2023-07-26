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
#   s.isnull().values.any()
#   s.isnull().sum()
#   s/df.notna()
#   s/df.isna()
#   df.isnull().sum()
#   df.isnull().sum().sum()
#   pd.isna(s)
import pandas as pd


def nan_replace(df: pd.DataFrame,
                fillna=None, fill_method=None, fill_limit=None,
                interpolate=None, interpolate_method=None, interpolate_order=None, interpolate_limit=None) -> pd.DataFrame:
    # method: ['pad', 'ffill', 'bfill', 'backfill'], limit: int
    # method: ['barycentric', 'quadratic', 'pchip', 'akima']
    # method: ['spline', 'polynomial'],              order: int, limit
    pass
