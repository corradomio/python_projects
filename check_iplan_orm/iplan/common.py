import typing
from datetime import datetime

import pandas as pd
from pandas import DataFrame

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


def start_end_dates(
    start_date: typing.Optional[datetime],
    end_date: typing.Optional[datetime],
    periods: typing.Optional[int],
    freq: typing.Literal['D', 'W', 'M'],
    which_date: typing.Optional[bool] = None
) -> tuple[typing.Optional[datetime], typing.Optional[datetime]]:
    """
    Compute the start/end dates based on the passed dates,
    the number of periods and the period frequency

    It is possible to specify which date to compute:

        - None: both dates
        - True: end_date
        - False: start_date

    :param start_date: start date
    :param end_date: end date
    :param periods: number of periods
    :param freq: period length (frequency)
    :param which_date: which date to compute
    :return:
    """
    assert is_instance(start_date, typing.Optional[datetime])
    assert is_instance(end_date, typing.Optional[datetime])
    assert is_instance(periods, typing.Optional[int])
    assert is_instance(freq, typing.Literal['D', 'W', 'M'])

    # not date specified: none to do
    if start_date is None and end_date is None:
        return None, None
    # both dates already specified: none to do
    if start_date is not None and end_date is not None:
        return start_date, end_date
    # start date specified and requires the start date: none to do
    if start_date is not None and which_date is False:
        return start_date, end_date
    # end date specified and it is required the end date: none to do
    if end_date is not None and which_date is True:
        return start_date, end_date

    # periods and freq MUST be specified
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
# concatenate_no_skill_df

def concatenate_no_skill_df(df_with_skill: DataFrame, df_no_skill: DataFrame, skill_features_dict: dict[int, str]):
    # this code implements the same logic implemented in 'replicateUnskilledMeasuresAgainstAllSkilledMeasures(...)'
    if len(df_no_skill) == 0:
        return df_with_skill

    logging.getLogger('ipom.om').error("Function 'concatenate_no_skill_df(...)' Not implemented yet")
    return df_with_skill


# def fill_missing_measures(df_pivoted: DataFrame, measure_dict: dict[int, str], new_format: bool) -> DataFrame:
#     measure_ids = measure_dict.keys()
#     # add missing columns (str(measure_id) OR measure_name) if necessary
#     for mid in measure_ids:
#         mname = measure_dict[mid] if new_format else str(mid)
#         if mname not in df_pivoted.columns:
#             df_pivoted[mname] = 0.
#     return df_pivoted


# ---------------------------------------------------------------------------
# fill_missing_dates

def fill_missing_dates(
    df_data: DataFrame, *,
    area_dict: dict[int, str],
    skill_dict: dict[int, str],
    measure_dict: dict[int, str],
    date_range) -> DataFrame:
    """
    Fill the dataframe with the missing values to reach the end_date
    There are two cases:

        1) it is specified end_date: used to expand the train data
        2) it is specified (start_date, periods): used to expand the predict data

    :param df_data: dataframe containing the data
    :param area_dict: areas to populate
    :param skill_dict: skills to populate
    :param measure_dict: measures to generate
    :return:
    """
    assert is_instance(df_data, DataFrame)
    # assert is_instance(area_dict, dict[int, str])
    # assert is_instance(skill_dict, dict[int, str])
    # assert is_instance(measure_dict, dict[int, str])

    # check the dataframe format
    new_format = 'area' in df_data.columns

    df_default = create_default_dataframe(
        date_range,
        area_dict=area_dict, skill_dict=skill_dict, measure_dict=measure_dict,
        new_format=new_format
    )

    df_merged = merge_dataframes(df_data, df_default)
    return df_merged


# ---------------------------------------------------------------------------
# create_default_dataframe
# merge_default_dataframe
# ---------------------------------------------------------------------------

def create_default_dataframe(
    date_range,
    area_dict: dict[int, str], skill_dict: dict[int, str], measure_dict: dict[int, str],
    new_format
):
    """
    Create the default dataframe using the
    :param start_date:
    :param periods:
    :param freq:
    :param area_dict:
    :param skill_dict:
    :param measure_dict:
    :return:
    """
    # assert is_instance(start_date, Union[datetime, pd.Timestamp])
    # assert is_instance(periods, int)
    # assert is_instance(freq, Literal['D', 'W', 'M'])
    # assert is_instance(area_dict, dict[int, str])
    # assert is_instance(skill_dict, dict[int, str])
    # assert is_instance(measure_dict, dict[int, str])

    periods = len(date_range)

    # compose the dataframe with for all areas/skill/daterange/measures
    df_list = []
    for area_id in area_dict:
        for skill_id in skill_dict:
            for measure_id in measure_dict:
                df_area_skill = DataFrame(index=list(range(periods)))
                df_area_skill['state_date'] = date_range.to_series().reset_index(drop=True)
                df_area_skill['area_id_fk'] = area_id
                df_area_skill['skill_id_fk'] = skill_id
                df_area_skill['model_detail_id_fk'] = measure_id
                df_area_skill['value'] = float(0)
                df_list.append(df_area_skill)
    # end
    df_flatten = pd.concat(df_list, axis=0, ignore_index=True)
    df_pivoted = pivot_df(df_flatten, area_dict, skill_dict, measure_dict, new_format=new_format)
    return df_pivoted


def merge_dataframes(df_data: DataFrame, df_default: DataFrame) -> DataFrame:
    assert is_instance(df_default, DataFrame)
    assert is_instance(df_data, DataFrame)

    if len(df_data) == 0:
        return df_default
    if len(df_default) == 0:
        return df_data

    df_merged = pd.concat([df_data, df_default]).reset_index(drop=True)
    return df_merged


# ---------------------------------------------------------------------------
# pivot_df
# compose_predict_df
# ---------------------------------------------------------------------------

# DON'T' TOUCH the order!
INDEX_COLUMNS = ['state_date', 'skill_id_fk', 'area_id_fk']
DATE_COLUMN = 'state_date'
VALUE_COLUMN = 'value'
PIVOT_COLUMNS = ['model_detail_id_fk']
FLATTEN_FORMAT_COLUMNS = INDEX_COLUMNS + PIVOT_COLUMNS + [VALUE_COLUMN]


def pivot_df(df: DataFrame,
             area_dict: dict[int, str], skill_dict: [int, str], measure_dict: [int, str],
             new_format: bool) -> DataFrame:

    # 0) check if all mandatory columns are present in the dataframe
    #    dataframe in 'flatten format'
    #   'area_id_fk', 'skill_id_fk', 'state_date', 'model_detail_id_fk', 'value'

    assert len(df.columns.intersection(FLATTEN_FORMAT_COLUMNS)) == len(FLATTEN_FORMAT_COLUMNS)

    # 1) transpose the dataframe
    df_pivoted = df.pivot_table(
        index=INDEX_COLUMNS,
        columns=PIVOT_COLUMNS,
        values=VALUE_COLUMN
    ).fillna(0)

    # 2.1) move the multiindex as columns
    df_pivoted.reset_index(inplace=True, names=INDEX_COLUMNS)

    # 2.2) force the datetime column values to be of type 'datetime'
    dtcol = df_pivoted[DATE_COLUMN]
    df_pivoted[DATE_COLUMN] = pd.to_datetime(dtcol)

    if new_format:
        # 'area', 'skill', 'date', <measure_name: str>, ...
        #         area & skill: string

        # 1) replace area/skill ids with names
        df_pivoted.replace(to_replace={
            'area_id_fk': area_dict,
            'skill_id_fk': skill_dict,
            'model_detail_id_fk': measure_dict
        }, inplace=True)

        # 2) rename the columns
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

        # # used to convert a column with an integer as name into a string
        # cdict = {mid: str(mid) for mid in measure_dict} | {
        #     'state_date': 'time'
        # }
        # df_pivoted.rename(columns=cdict, inplace=True)
        #
        # # add the column 'day', based on the column 'time'
        # df_pivoted['day'] = df_pivoted['time'].dt.day_name()

        #
        # BETTER: KEEP PIVOTED FORMAT
        #
        pass
    # end
    return df_pivoted


# ---------------------------------------------------------------------------
# normalize_df
# ---------------------------------------------------------------------------
# To see 'README_dataframe_format
#

def normalize_df(df: DataFrame,
                 area_features_dict: dict[int, str],
                 skill_features_dict: dict[int, str],
                 measure_dict: dict[int, str],
                 freq: typing.Literal['D', 'W', 'M']) -> DataFrame:
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
        Pandas 'timestamp' or 'period' or Python 'datetime' ???

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
    # 2.2) collect the data columns as integer, integer as string, name
    for mid in measure_dict:
        # add measure_id as int, str and name (str)
        mandatory_columns += [mid, str(mid), measure_dict[mid]]

    # 2.2) identifies the extra columns and drop then
    extra_columns = df.columns.difference(mandatory_columns)
    if len(extra_columns) > 0:
        df.drop(labels=extra_columns, axis=1, inplace=True)

    measure_ids = list(measure_dict.keys())

    area_features_drev = reverse_dict(area_features_dict)
    skill_drev = reverse_dict(skill_features_dict)
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
            'skill': skill_drev
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


# ---------------------------------------------------------------------------
# where_start_end_date
# ---------------------------------------------------------------------------

def where_start_end_date(
    table, query, *,
    start_date: typing.Optional[datetime], end_date: typing.Optional[datetime],
    dtcol: str = 'state_date',
    start_included: bool = True,
    end_included: bool = False,
):
    """
    Add optional conditions on start/end dates

    :param table: table to use
    :param dtcol: datetime column name
    :param query: query to use
    :param start_date: optional start date
    :param end_date: optional end date
    :param start_included: if to include the start date
    :param end_included: if to include the end date
    :return: updated query
    """

    assert is_instance(dtcol, str)
    assert is_instance(start_date, typing.Optional[datetime])
    assert is_instance(end_date, typing.Optional[datetime])
    assert is_instance(start_included, bool)
    assert is_instance(end_included, bool)

    if start_date is not None:
        if start_included:
            query = query.where(table.c[dtcol] >= start_date)
        else:
            query = query.where(table.c[dtcol] > start_date)
    if end_date is not None:
        if end_included:
            query = query.where(table.c[dtcol] <= end_date)
        else:
            query = query.where(table.c[dtcol] < end_date)
    return query
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------

