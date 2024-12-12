from typing import Union
from stdlib import as_list, is_instance
import pandas as pd
from pandasx import groups_split, groups_merge


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

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


def update(df: pd.DataFrame, df_update: pd.DataFrame,
           select: str,
           update: Union[str, list[str]],
           groups: Union[None, str, list[str]] = None,
           inplace=False) -> pd.DataFrame:
    """
    Update the columns 'update' of 'df' using the content of 'udf'.


    :param df: dataframe to update
    :param df_update: dataframe containing the values to use for the updating
    :param select: column used to select the rows to update
    :param update: column or list of columns to update
    :param inplace: if to update inplace
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_update, pd.DataFrame)
    assert isinstance(select, str)
    assert is_instance(update, Union[str, list[str]])

    if not inplace:
        df = df.copy()

    columns = as_list(update)

    if groups is None and not isinstance(df.index, pd.MultiIndex):
        return _update_single(df, df_update, select, columns)

    df_dict = groups_split(df, groups=groups)
    udf_dict = groups_split(df_update, groups=groups)

    upd_dict = {}
    for g in udf_dict:
        if g not in df_dict:
            continue

        dfg = df_dict[g]
        udfg = udf_dict[g]

        updg = _update_single(dfg, udfg, select, columns)

        upd_dict[g] = updg
    # end

    updated = groups_merge(upd_dict, groups=groups)

    return updated
# end


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------

def _merge_single(df: pd.DataFrame, to_merge: pd.DataFrame):

    # we suppose that the column 'select' has a total order and there is a
    # correspondence of 1:1 with the rows in 'udf'
    # special handling for 'datetime' columns!

    m_index = df.index.intersection(to_merge.index)
    if len(m_index) == 0:
        return df
    if len(to_merge) == 0:
        return df
    if len(df) == 0:
        return to_merge

    for col in df.columns:
        df[col][m_index] = to_merge[col][m_index]

    return df


def merge(df: pd.DataFrame, to_merge: pd.DataFrame,
          groups: Union[None, str, list[str]] = None) -> pd.DataFrame:
    """
    Update the columns 'update' of 'df' using the content of 'udf'.


    :param df: dataframe to update
    :param udf: dataframe containing the values to use for the updating
    :param select: column used to select the rows to update
    :param update: column or list of columns to update
    :param inplace: if to update inplace
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(to_merge, pd.DataFrame)

    if groups is None and not isinstance(df.index, pd.MultiIndex):
        return _merge_single(df, to_merge)

    df_dict = groups_split(df, groups=groups)
    to_merge_dict = groups_split(to_merge, groups=groups)

    merge_dict = {}
    for g in df_dict:
        if g not in to_merge_dict:
            merge_dict[g] = df_dict[g]
            continue

        dfg = df_dict[g]
        to_mergeg = to_merge_dict[g]

        mergedg = _merge_single(dfg, to_mergeg)

        merge_dict[g] = mergedg
    # end

    merged = groups_merge(merge_dict, groups=groups)

    return merged
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
