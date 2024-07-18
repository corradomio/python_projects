from typing import Union
from stdlib import as_list, is_instance
import pandas as pd
from pandasx import groups_split, groups_merge


def _update_single(df: pd.DataFrame, udf: pd.DataFrame, select: str, columns: list[str]):

    # we suppose that the column 'select' has a total order and there is a
    # correspondence of 1:1 with the rows in 'udf'
    # special handling for 'datetime' columns!

    min_val = udf[select].min()
    max_val = udf[select].max()

    index = df[(df[select] >= min_val) & (df[select] <= max_val)].index
    if len(index) != len(udf):
        raise ValueError("Unable update the dataframe: the number of rows is not the same")

    udf = udf.set_index(index, inplace=False)

    for col in columns:
        df[col] = udf[col]

    return df


def update(df: pd.DataFrame, udf: pd.DataFrame, select: str, update: Union[str, list[str]],
           groups: Union[None, str, list[str]] = None,
           inplace=False) -> pd.DataFrame:
    """
    Update the columns 'update' of 'df' using the content of 'udf'.


    :param df: dataframe to update
    :param udf: dataframe containing the values to use for the updating
    :param select: column used to select the rows to update
    :param update: column or list of columns to update
    :param inplace: if to update inplace
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(udf, pd.DataFrame)
    assert isinstance(select, str)
    assert is_instance(update, Union[str, list[str]])

    if not inplace:
        df = df.copy()

    columns = as_list(update)

    if groups is None and not isinstance(df.index, pd.MultiIndex):
        return _update_single(df, udf, select, columns)

    df_dict = groups_split(df, groups=groups)
    udf_dict = groups_split(udf, groups=groups)

    upd_dict = {}
    for g in udf_dict:
        if g not in df_dict:
            continue

        dfg = df_dict[g]
        udfg = udf_dict[g]

        updg = _update_single(dfg, udfg, select, columns)

        upd_dict[g] = updg
    # end

    upd = groups_merge(upd_dict, groups=groups)

    return upd
# end
