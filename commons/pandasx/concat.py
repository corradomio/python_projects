from typing import Optional

import pandas as pd
from pandas import DataFrame
from pandasx import groups_split, groups_merge

# objs: Iterable[Series | DataFrame] | Mapping[HashableT, Series | DataFrame],
#     *,
#     axis: Axis = 0,
#     join: str = "outer",
#     ignore_index: bool = False,
#     keys: Iterable[Hashable] | None = None,
#     levels=None,
#     names: list[HashableT] | None = None,
#     verify_integrity: bool = False,
#     sort: bool = False,
#     copy: bool | None = None,


def _concat_single(df_list:list[DataFrame], select:Optional[str]) -> DataFrame:
    # 1) set the index using select
    df_list = [
        df.set_index(keys=select, drop=False)
        for df in df_list
    ]
    df0 = df_list[0]

    all_index = df0.index
    for df in df_list:
        all_index = all_index.union(df.index)

    columns = list(df0.columns)

    df_concat = DataFrame(columns=df0.columns, index=all_index)

    for df in df_list:
        for col in columns:
            index = df.index
            df_concat.loc[index, col] = df[col]

    # 2) concat using the index
    # df_concat = pd.concat(df_list, ignore_index=False)
    # df_concat.sort_index(inplace=True)
    df_concat.reset_index(drop=True, inplace=True)
    return df_concat
# end


def concat(
    df_list: list[DataFrame],
    select:Optional[str]=None,
    groups:Optional[list[str]]=None
) -> DataFrame:
    # extends concat to concatenate multiple dataframes based on datetime
    if select is None:
        return pd.concat(df_list)

    if groups is None and not isinstance(df_list[0].index, pd.MultiIndex):
        return _concat_single(df_list, select)

    df_dict_list = [
        groups_split(df, groups=groups)
        for df in df_list
    ]

    concat_dict = {}

    keys = list(df_dict_list[0].keys())
    for k in keys:
        dfk_list = [df_dict[k] for df_dict in df_dict_list]
        dfk_cat = _concat_single(dfk_list, select=select)
        concat_dict[k] = dfk_cat
    # end

    df_concat = groups_merge(concat_dict, groups=groups)
    return df_concat
# end
