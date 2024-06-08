from datetime import datetime
from typing import Union, Any, Literal, Optional

import pandas as pd
from pandas import DataFrame

import pandasx as pdx
import stdlib.loggingx as logging
from stdlib.dateutilx import relativeperiods
from stdlib.dict import reverse_dict
from stdlib.is_instance import is_instance


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def safe_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def start_end_dates(start_date: Optional[datetime], end_date: Optional[datetime], periods: Optional[int],
                    freq: Literal['D', 'W', 'M']) \
    -> tuple[Optional[datetime], Optional[datetime]]:
    assert is_instance(start_date, Optional[datetime])
    assert is_instance(end_date, Optional[datetime])
    assert is_instance(periods, Optional[int])
    assert is_instance(freq, Literal['D', 'W', 'M'])

    if start_date is None and end_date is None:
        return None, None
    if start_date is not None and end_date is not None:
        return start_date, end_date
    if periods is None or periods <= 0 or freq is None:
        raise ValueError("Unable to resolve start_date/end_date: missing periods or frequency")

    if end_date is None:
        end_date = start_date + relativeperiods(periods=periods, freq=freq)
    elif start_date is None:
        start_date = end_date - relativeperiods(periods=periods, freq=freq)
    else:
        raise ValueError("Unable to resolve start_date/end_date: missing periods or frequency")

    return start_date, end_date


# ---------------------------------------------------------------------------
# Pandas utilities
# ---------------------------------------------------------------------------

def concatenate_no_skill_df(df_with_skill: DataFrame, df_no_skill: DataFrame, skill_features_dict: dict[int, str]):
    # this code implements the same logic implemented in 'replicateUnskilledMeasuresAgainstAllSkilledMeasures(...)'
    if len(df_no_skill) == 0:
        return df_with_skill

    logging.getLogger('ipom.om').error("Function 'concatenate_no_skill_df(...)' Not implemented yet")
    return df_with_skill


def fill_missing_measures(df_pivoted: DataFrame, measure_dict: dict[int, str], new_format: bool) -> DataFrame:
    measure_ids = measure_dict.keys()
    # add missing columns (str(measure_id) OR measure_name) if necessary
    for mid in measure_ids:
        mname = measure_dict[mid] if new_format else str(mid)
        if mname not in df_pivoted.columns:
            df_pivoted[mname] = 0.
    return df_pivoted


def fill_missing_dates(df_pivoted: DataFrame, start_date: datetime, end_date: datetime) -> DataFrame:
    return df_pivoted


# ---------------------------------------------------------------------------
# pivot_df
# compose_predict_df
# ---------------------------------------------------------------------------

def pivot_df(df: DataFrame,
             area_feature_dict: dict[int, str], skill_feature_dict: [int, str], measure_dict: [int, str],
             new_format: bool) -> DataFrame:
    # 0) check if all mandatory columns are present in the dataframe

    INDEX_COLUMNS = ['state_date', 'skill_id_fk', 'area_id_fk']
    DATE_COLUMN = 'state_date'
    VALUE_COLUMN = 'value'
    PIVOT_COLUMNS = ['model_detail_id_fk']
    DATA_MANDATORY_COLUMNS = INDEX_COLUMNS + PIVOT_COLUMNS + [VALUE_COLUMN]

    assert len(df.columns.intersection(DATA_MANDATORY_COLUMNS)) == len(DATA_MANDATORY_COLUMNS)

    # 1) transpose the dataframe
    df_pivoted = df.pivot_table(
        index=INDEX_COLUMNS,
        columns=PIVOT_COLUMNS,
        values=VALUE_COLUMN
    ).fillna(0)

    # 2) move the multiindex as columns
    #    force the date column values to be of type 'datetime'
    df_pivoted.reset_index(inplace=True, names=INDEX_COLUMNS)
    df_pivoted[DATE_COLUMN] = pd.to_datetime(df_pivoted[DATE_COLUMN])

    if new_format:
        # replace area/skill ids with names
        df_pivoted.replace(to_replace={
            'area_id_fk': area_feature_dict,
            'skill_id_fk': skill_feature_dict,
            'model_detail_id_fk': measure_dict
        }, inplace=True)

        # rename the columns
        df_pivoted.rename(columns=measure_dict | {
            'area_id_fk': 'area',
            'skill_id_fk': 'skill',
            'state_date': 'date'
        }, inplace=True)
        pass
    else:
        # COMPATIBILITY with the original format

        # ensure all column names as string
        # 'area_id_fk'
        # 'skill_id_fk'
        # 'state_date' -> 'time'
        # measure_id: int

        # used to convert a column with an integer as name into a string
        cdict = {mid: str(mid) for mid in measure_dict} | {
            'state_date': 'time'
        }
        df_pivoted.rename(columns=cdict, inplace=True)

        # add the column 'day', based on the column 'time'
        df_pivoted['day'] = df_pivoted['time'].dt.day_name()
        pass
    # end
    return df_pivoted


# def compose_predict_df(df_past: DataFrame,
#                        area_feature_dict: dict[int, str],
#                        skill_feature_dict: dict[int, str],
#                        input_measure_ids: list[int],
#                        measure_dict: dict[int, str],
#                        start_end_dates: dict[int, tuple[datetime, datetime]],
#                        freq: Literal['D', 'W', 'M'],
#                        defval: Union[None, float] = 0.,
#                        new_format=False) -> DataFrame:
#     """
#
#     :param df_past: dataframe to process
#     :param area_feature_dict: dictionary area id->name
#     :param skill_feature_dict: dictionary skill id->name
#     :param input_measure_ids: list of measures used as input features
#     :param measure_dict: dictionary measure id->name
#     :param start_end_dates: dictionary start/end dates for each area
#     :param freq: period frequency (daily, weekly, monthly)
#     :param defval: default value to use for filling
#     :param new_format: dataset format (old/new)
#     :return: processed dataset
#     """
#     assert is_instance(df_past, DataFrame)
#     assert is_instance(area_feature_dict, dict[int, str])
#     assert is_instance(skill_feature_dict, dict[int, str])
#     assert is_instance(input_measure_ids, list[int])
#     assert is_instance(measure_dict, dict[int, str])
#     assert is_instance(start_end_dates, dict[int, tuple[datetime, datetime]])
#     assert is_instance(freq, Literal['D', 'W', 'M'])
#     assert is_instance(defval, Union[None, float])
#
#     df_future = _create_df_future(
#         df_past,
#         area_feature_dict, skill_feature_dict,
#         input_measure_ids, measure_dict,
#         start_end_dates, freq, defval,
#         new_format=new_format)
#
#     df_future = _merge_df_past_future(
#         df_past, df_future,
#         input_measure_ids, measure_dict,
#         new_format=new_format)
#     return df_future


# def _create_df_future(
#     df_past: DataFrame,
#     area_feature_dict: dict[int, str],
#     skill_feature_dict: dict[int, str],
#     input_measure_ids: list[int],
#     measure_dict: dict[int, str],
#     start_end_dates: dict[int, tuple[datetime, datetime]],
#     freq: Literal['D', 'W', 'M'],
#     defval: Union[None, float] = 0.,
#     new_format=False
# ) -> DataFrame:
#     if new_format:
#         df_future = _create_df_future_new_format(
#             df_past,
#             area_feature_dict,
#             skill_feature_dict,
#             input_measure_ids,
#             measure_dict,
#             start_end_dates, freq,
#             defval)
#
#         # return _compose_predict_df_new_format(
#         #     df_past,
#         #     area_feature_dict,
#         #     skill_feature_dict,
#         #     input_measure_ids,
#         #     measure_dict,
#         #     start_end_dates, freq,
#         #     defval)
#     else:
#         df_future = _create_df_future_old_format(
#             df_past,
#             area_feature_dict,
#             skill_feature_dict,
#             input_measure_ids,
#             measure_dict,
#             start_end_dates, freq,
#             defval)
#
#     return df_future


# def _create_df_future_new_format(
#     df_past: DataFrame,
#     area_feature_dict: dict[int, str],
#     skill_feature_dict: dict[int, str],
#     input_measure_ids: list[int],
#     measure_dict: dict[int, str],
#     start_end_dates: dict[int, tuple[datetime, datetime]],
#     freq: Literal['D', 'W', 'M'],
#     defval: Union[None, float] = 0.) -> DataFrame:
#     area_feature_drev = reverse_dict(area_feature_dict)
#     skill_feature_drev = reverse_dict(skill_feature_dict)
#     measure_drev = reverse_dict(measure_dict)
#
#     # columns: ['area', 'skill', 'date', <measure_name>]
#
#     area_skill_list = pdx.groups_list(df_past, groups=['area', 'skill'])
#     df_future_list = []
#     for area_skill in area_skill_list:
#         area_name, skill_name = area_skill
#
#         # check if area_name, skill_name are defined
#         if area_name not in area_feature_drev or skill_name not in skill_feature_drev:
#             continue
#
#         area_feature_id = area_feature_drev[area_name]
#         # skill_feature_id = skill_feature_drev[skill_name]
#
#         # trick: 0 is used as DEFAULT datetime for all areas
#         if 0 in start_end_dates:
#             start_date, end_date = start_end_dates[0]
#         elif area_feature_id in start_end_dates:
#             start_date, end_date = start_end_dates[area_feature_id]
#         else:
#             # no start/end dates is found
#             continue
#
#         #  FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
#         #   MS: MonthBegin
#         #   ME: MonthEnd
#         if freq == 'M': freq = 'MS'
#
#         date_index = pd.date_range(start=start_date, end=end_date, freq=freq)
#         # date_index = pd.period_range(start=start_date, end=end_date, freq=freq)
#
#         # create the dataframe
#         area_skill_df = DataFrame(data={
#             'area': area_name,
#             'skill': skill_name,
#             'date': date_index.to_series()
#         })
#
#         # fill the dataframe with the measures
#         for measure_name in measure_drev:
#             area_skill_df[measure_name] = defval
#
#         df_future_list.append(area_skill_df)
#         pass
#     # end
#     df_future = pd.concat(df_future_list, axis=0, ignore_index=True).reset_index(drop=True)
#     return df_future


def _create_df_future_old_format(
    df_past: DataFrame,
    area_feature_dict: dict[int, str],
    skill_feature_dict: dict[int, str],
    input_measure_ids: list[int],
    measure_dict: dict[int, str],
    start_end_dates: dict[int, tuple[datetime, datetime]],
    freq: Literal['D', 'W', 'M'],
    defval: Union[None, float] = 0.) -> DataFrame:
    # columns: ['skill_id_fk', 'area_id_fk', 'time', 'day', <measure_id: str>]

    area_skill_list = pdx.groups_list(df_past, groups=['area_id_fk', 'skill_id_fk'])
    df_future_list = []
    for area_skill in area_skill_list:
        area_feature_id = int(area_skill[0])
        skill_feature_id = int(area_skill[1])

        # check if area_name, skill_name are defined
        if area_feature_id not in area_feature_dict or skill_feature_id not in skill_feature_dict:
            continue

        # trick: 0 is used as DEFAULT datetime for all areas
        if 0 in start_end_dates:
            start_date, end_date = start_end_dates[0]
        elif area_feature_id in start_end_dates:
            start_date, end_date = start_end_dates[area_feature_id]
        else:
            # no start/end dates is found
            continue

        #  FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
        #   MS: MonthBegin
        #   ME: MonthEnd
        if freq == 'M': freq = 'MS'

        date_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        # date_index = pd.period_range(start=start_date, end=end_date, freq=freq)

        # create the dataframe
        area_skill_df = DataFrame(data={
            'area_id_fk': area_feature_id,
            'skill_id_fk': skill_feature_id,
            'time': date_index.to_series()
        })
        area_skill_df['day'] = area_skill_df['time'].dt.day_name()

        # fill the dataframe with the measures (measure_id as string)
        for measure_id in measure_dict:
            area_skill_df[str(measure_id)] = defval

        df_future_list.append(area_skill_df)
        pass
    # end

    df_future = pd.concat(df_future_list, axis=0, ignore_index=True).reset_index(drop=True)
    return df_future


def _merge_df_past_future(df_past: DataFrame, df_future: DataFrame,
                          input_measure_ids: list[int], measure_dict: dict[int, str],
                          new_format=True) -> DataFrame:
    assert is_instance(df_past, DataFrame)
    assert is_instance(df_future, DataFrame)
    assert is_instance(input_measure_ids, list[int])
    assert is_instance(measure_dict, dict[int, str])
    assert sorted(df_past.columns) == sorted(df_future.columns)

    groups = ['area', 'skill'] if new_format else ['area_id_fk', 'skill_id_fk']
    date_col = 'date' if new_format else 'time'

    df_past_dict = pdx.groups_split(df_past, groups=groups)
    df_future_dict = pdx.groups_split(df_future, groups=groups)
    df_predict_dict = {}

    for group in df_future_dict:
        if group not in df_past_dict:
            continue

        dfp = df_past_dict[group]
        dff = df_future_dict[group]

        past_min = dfp[date_col].min()
        past_max = dfp[date_col].max()
        future_min = dff[date_col].min()
        future_max = dff[date_col].max()

        if past_max < future_min:
            df_predict_dict[group] = dff
        elif past_max >= future_max:
            continue
        elif past_min < future_min < past_max:
            df_predict_dict[group] = dff[dff[date_col] > past_min]
        else:
            continue
    # end

    df_pred = pdx.groups_merge(df_predict_dict, groups=groups)
    df_merged = pd.concat([df_pred, df_future], axis=0, ignore_index=True)
    return df_merged


def _compose_predict_df_old_format(
    df_past: DataFrame,
    area_feature_dict: dict[int, str],
    skill_feature_dict: dict[int, str],
    input_measure_ids: list[int],
    measure_dict: dict[int, str],
    start_end_dates: dict[int, tuple[datetime, datetime]],
    freq: Literal['D', 'W', 'M'],
    defval: Union[None, float] = 0.) -> DataFrame:
    # columns: ['skill_id_fk', 'area_id_fk', 'time', 'day', <measure_id>]

    measure_ids = set(measure_dict.keys())
    target_measure_ids = measure_ids.difference(input_measure_ids)

    df_list = []
    for area_feature_id in area_feature_dict:
        start_date, end_date = None, None

        # trick: 0 is used as DEFAULT datetime for all areas
        if 0 in start_end_dates:
            start_date, end_date = start_end_dates[0]
        if area_feature_id in start_end_dates:
            start_date, end_date = start_end_dates[area_feature_id]
        if start_date is None:
            # no start/end dates is found
            continue

        date_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        for skill_feature_id in skill_feature_dict:
            area_skill_df = DataFrame(data={
                'area_id_fk': area_feature_id,
                'skill_id_fk': skill_feature_id,
                'time': date_index.to_series()
            })
            area_skill_df['day'] = area_skill_df['time'].dt.day_name()
            for input_id in input_measure_ids:
                area_skill_df[str(input_id)] = defval

            for target_id in target_measure_ids:
                area_skill_df[str(target_id)] = defval

            df_list.append(area_skill_df)

    df_pred = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)
    return df_pred


def _compose_predict_df_new_format(
    df_past: DataFrame,
    area_feature_dict: dict[int, str],
    skill_feature_dict: dict[int, str],
    input_measure_ids: list[int],
    measure_dict: dict[int, str],
    start_end_dates: dict[int, tuple[datetime, datetime]],
    freq: Literal['D', 'W', 'M'],
    defval: Union[None, float] = 0.) -> DataFrame:
    # columns: ['area', 'skill', 'date', <measure_name>]

    measure_ids = set(measure_dict.keys())
    target_measure_ids = measure_ids.difference(input_measure_ids)

    df_list = []
    for area_feature_id in area_feature_dict:
        start_date, end_date = None, None

        # trick: 0 is used as DEFAULT datetime for all areas
        if 0 in start_end_dates:
            start_date, end_date = start_end_dates[0]
        if area_feature_id in start_end_dates:
            start_date, end_date = start_end_dates[area_feature_id]
        if start_date is None:
            # no start/end dates is found
            continue

        #  FutureWarning: 'M' is deprecated and will be removed in a future version, please use 'ME' instead.
        #   MS: MonthBegin
        #   ME: MonthEnd
        if freq == 'M': freq = 'MS'

        date_index = pd.date_range(start=start_date, end=end_date, freq=freq)
        for skill_feature_id in skill_feature_dict:
            area_skill_df = DataFrame(data={
                'area': area_feature_dict[area_feature_id],
                'skill': skill_feature_dict[skill_feature_id],
                'date': date_index.to_series()
            })
            for input_id in input_measure_ids:
                input_name = measure_dict[input_id]
                area_skill_df[input_name] = defval
            for target_id in target_measure_ids:
                target_name = measure_dict[target_id]
                area_skill_df[target_name] = defval

            df_list.append(area_skill_df)

    df_pred = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)
    return df_pred


# ---------------------------------------------------------------------------
# normalize_df
# ---------------------------------------------------------------------------
# df can have the following formats
#
# old format:
#   index: not used
#   'area_id_fk':     integer values
#   'skill_id_fk':    integer values
#   'time':           datetime
#   'day':            str
#   <measure_id>:     float values
#   ...
#
#   Note: <measure_id> can be an integer or a string
#
# old_format/multiindex
#   index: ['area_id_fk'/'skill_id_fk'/'time']
#   'day':            str
#   <measure_id>:     float values
#   ...
#
#   Note: <measure_id> can be an integer or a string
#
# new format:
#   index: not used
#   'area':           string values
#   'skill':          string values
#   'date':           datetime
#   <measure_name>:   float values
#   ...
#
# new format/multiindex:
#   index: ['area'/'skill'/'date']
#   measure_name:   float values
#
# normalized format:
#   'area_id_fk': int values
#   'skill_id_fk': int values
#   'state_date": timestamp
#   <measure_id>: column as integer, float values
#

def normalize_df(df: DataFrame,
                 area_features_dict: dict[int, str],
                 skill_features_dict: dict[int, str],
                 measure_dict: dict[int, str],
                 freq: Literal['D', 'W', 'M']) -> DataFrame:
    """
    Normalize the dataframe 'df'
    There are two possible dataframe formats:

    new format with columns:

        'area', 'skill', 'date', <measure_name:str>, ...
        with area and skill specified by name and 'date' as

    old format with columns:

        'area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id: str|int>, ...
        with area and skill specified by id:int
        the column 'day' is optional, it is not considered, because it contains
        the same information available in 'time'

    The 'normalized format' has columns

        'area_id_fk', 'skill_id_fk', 'state_date', <measure_id: int>, ...
        with area and skill specified by id:int

    The dataframe can contain a 'multiindex', BUT, when removed, it MUST generate
    the columns:

        'area', 'skill', 'date'

    or

        'area_id_fk', 'skill_id_fk', 'time'

    The columns ['area', 'skill', 'date'] or ['area_id_fk', 'skill_id_fk', 'time']
    are mandatory.

    If 'df' has more columns than the columns specified in 'measure_dict'
    the extra columns will be removed. If there are less columns, it is raised
    an exception.

    WARN: HOW TO SPECIFY the timestamp?
        Pandas 'timestamp' or 'period'

    :param df: dataframe to process
    :param area_features_dict:  dict area_id->area_name
    :param skill_features_dict: dict skill_id->skill_name
    :param measure_dict:        dict measure_id->measure_name
    :param freq: period frequency
    :return: a dataframe normalized
    """
    log = logging.getLogger("iplan.normalize_df")

    # 1) remove the multi index
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.copy()

    # Note: the dataframe can contains MORE columns than the list of columns
    #       absolutely necessary. It can be a good idea to remove them

    # 2.1) collect the mandatory columns:
    mandatory_columns = ['area', 'skill', 'date', 'area_id_fk', 'area_id_fk', 'time']
    for mid in measure_dict:
        # add measure_id (int|str) and measure_name (str)
        mandatory_columns += [mid, str(mid), measure_dict[mid]]

    # 2.2) identifies the extra columns and drop then
    extra_columns = df.columns.difference(mandatory_columns)
    if len(extra_columns) > 0:
        df.drop(labels=extra_columns, axis=1, inplace=True)

    measure_ids = list(measure_dict.keys())

    area_features_drev = reverse_dict(area_features_dict)
    skill_feature_drev = reverse_dict(skill_features_dict)
    measure_drev = reverse_dict(measure_dict)

    # 3) check the df structure: ensures that the format is correct
    columns = set(df.columns)
    if 'area_id_fk' in columns:
        if 'day' in columns:
            df.drop(labels=['day'], axis=1, inplace=True)
            columns = set(df.columns)

        # old format:
        # convert the column names, specified as an integer in string format, in
        # a real integer
        cnames = ['area_id_fk', 'skill_id_fk', 'time', 'day']
        for mid in measure_ids:
            cnames.extend([mid, str(mid)])

        if len(columns.intersection(cnames)) != 3 + len(measure_ids):
            raise ValueError(f"invalid dataFrame: columns={columns}, required={cnames}")
        new_format = False
    else:
        # new format
        cnames = ['area', 'skill', 'date']
        for mid in measure_ids:
            cnames.append(measure_dict[mid])
        if len(columns.intersection(cnames)) != 3 + len(measure_ids):
            log.error(f"Invalid dataFrame:")
            log.error(f" dataframe columns: ={columns}")
            log.error(f"  required columns: ={cnames}")
            log.error(f"  missing columns: {set(cnames).difference(columns)}")
            log.error(f"    extra columns: {set(columns).difference(cnames)}")
            raise ValueError(f"invalid dataFrame: columns={columns}, required={cnames}")
        new_format = True

    # 4) rename the column names & area/skill values
    if new_format:
        df.replace(to_replace={
            'area': area_features_drev,
            'skill': skill_feature_drev
        }, inplace=True)
        df.rename(columns=measure_drev | {
            'area': 'area_id_fk',
            'skill': 'skill_id_fk',
            'date': 'state_date',
        }, inplace=True)
    else:
        df.rename(columns=measure_drev | {
            'time': 'state_date',
        }, inplace=True)

    return df

