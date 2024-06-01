import stdlib.loggingx as logging
import traceback
from deprecated import deprecated
from stdlib import as_list
from stdlib.is_instance import is_instance
from stdlib.dateutilx import relativeperiods
from stdlib.dict import reverse_dict
import pandas as pd
import pandasx as pdx
from pandas import DataFrame
from typing import Optional, Union, Any, Literal, Self
from datetime import datetime, timedelta, date, time

from sqlalchemy import MetaData, Engine, Table, Row, create_engine, URL
from sqlalchemy import select, delete, insert, update, text, func


CREATED_BY = "Python IPlanObjectModel"


# ---------------------------------------------------------------------------
# to_data
# ---------------------------------------------------------------------------

def to_data(result: Row) -> dict[str, Any]:
    values = result._data
    fields = result._fields
    assert len(values) == len(fields)
    return {
        fields[i]: values[i] for i in range(len(result))
    }


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


def safe_int(s):
    try:
        return int(s)
    except ValueError:
        return s


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


def compose_predict_df(df_past: DataFrame,
                       area_feature_dict: dict[int, str],
                       skill_feature_dict: dict[int, str],
                       input_measure_ids: list[int],
                       measure_dict: dict[int, str],
                       start_end_dates: dict[int, tuple[datetime, datetime]],
                       freq: Literal['D', 'W', 'M'],
                       defval: Union[None, float] = 0.,
                       new_format=False) -> DataFrame:
    """

    :param df_past: dataframe to process
    :param area_feature_dict: dictionary area id->name
    :param skill_feature_dict: dictionary skill id->name
    :param input_measure_ids: list of measures used as input features
    :param measure_dict: dictionary measure id->name
    :param start_end_dates: dictionary start/end dates for each area
    :param freq: period frequency (daily, weekly, monthly)
    :param defval: default value to use for filling
    :param new_format: dataset format (old/new)
    :return: processed dataset
    """
    assert is_instance(df_past, DataFrame)
    assert is_instance(area_feature_dict, dict[int, str])
    assert is_instance(skill_feature_dict, dict[int, str])
    assert is_instance(input_measure_ids, list[int])
    assert is_instance(measure_dict, dict[int, str])
    assert is_instance(start_end_dates, dict[int, tuple[datetime, datetime]])
    assert is_instance(freq, Literal['D', 'W', 'M'])
    assert is_instance(defval, Union[None, float])

    df_future = _create_df_future(df_past,
                                  area_feature_dict, skill_feature_dict,
                                  input_measure_ids, measure_dict,
                                  start_end_dates, freq, defval,
                                  new_format=new_format)

    df_past, df_future = _merge_df_past_future(df_past, df_future,
                                               input_measure_ids, measure_dict,
                                               new_format=new_format)
    return df_past, df_future
# end


def _create_df_future(
        df_past: DataFrame,
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        input_measure_ids: list[int],
        measure_dict: dict[int, str],
        start_end_dates: dict[int, tuple[datetime, datetime]],
        freq: Literal['D', 'W', 'M'],
        defval: Union[None, float] = 0.,
        new_format=False
) -> DataFrame:

    if new_format:
        df_future = _create_df_future_new_format(
            df_past,
            area_feature_dict,
            skill_feature_dict,
            input_measure_ids,
            measure_dict,
            start_end_dates, freq,
            defval)

        # return _compose_predict_df_new_format(
        #     df_past,
        #     area_feature_dict,
        #     skill_feature_dict,
        #     input_measure_ids,
        #     measure_dict,
        #     start_end_dates, freq,
        #     defval)
    else:
        df_future = _create_df_future_old_format(
            df_past,
            area_feature_dict,
            skill_feature_dict,
            input_measure_ids,
            measure_dict,
            start_end_dates, freq,
            defval)

    return df_future
# end


def _create_df_future_new_format(
    df_past: DataFrame,
    area_feature_dict: dict[int, str],
    skill_feature_dict: dict[int, str],
    input_measure_ids: list[int],
    measure_dict: dict[int, str],
    start_end_dates: dict[int, tuple[datetime, datetime]],
    freq: Literal['D', 'W', 'M'],
    defval: Union[None, float] = 0.) -> DataFrame:

    area_feature_drev = reverse_dict(area_feature_dict)
    skill_feature_drev = reverse_dict(skill_feature_dict)
    measure_drev = reverse_dict(measure_dict)

    # columns: ['area', 'skill', 'date', <measure_name>]

    area_skill_list = pdx.groups_list(df_past, groups=['area', 'skill'])
    df_future_list = []
    for area_skill in area_skill_list:
        area_name, skill_name = area_skill

        # check if area_name, skill_name are defined
        if area_name not in area_feature_drev or skill_name not in skill_feature_drev:
            continue

        area_feature_id = area_feature_drev[area_name]
        # skill_feature_id = skill_feature_drev[skill_name]

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
            'area': area_name,
            'skill': skill_name,
            'date': date_index.to_series()
        })

        # fill the dataframe with the measures
        for measure_name in measure_drev:
            area_skill_df[measure_name] = defval

        df_future_list.append(area_skill_df)
        pass
    # end

    df_future = pd.concat(df_future_list, axis=0, ignore_index=True).reset_index(drop=True)
    return df_future
# end


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
# end


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
                 measure_dict: dict[int, str]) -> DataFrame:

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
        # old format
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


# ---------------------------------------------------------------------------
#   IPlanObject
#       IPlanData
# ---------------------------------------------------------------------------

class IPlanObject:

    def __init__(self, ipom: "IPlanObjectModel"):
        self.ipom: "IPlanObjectModel" = ipom
        self.engine = ipom.engine

        self.log = logging.getLogger(f"iplan.om.{self.__class__.__name__}")
    # end
# end


class IPlanData(IPlanObject):
    def __init__(self, ipom: "IPlanObjectModel", id: Union[int, dict], table: Table, idcol="id"):
        super().__init__(ipom)

        if isinstance(id, int):
            self._table = table
            self._idcol = idcol
            self._id = id
            self._data: dict[str, Any] = {}
        elif isinstance(id, dict):
            self._data = id
            self._table = table
            self._idcol = idcol
            self._id = self.data[idcol]
        else:
            raise ValueError(f"Unsupported {id}")

        self.log = logging.getLogger(f"iplan.om.{self.__class__.__name__}")

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        ...

    @property
    def idcol(self) -> str:
        return self._idcol

    @property
    def table(self) -> Table:
        return self._table

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    def check_data(self):
        if len(self._data) > 0:
            return

        with self.engine.connect() as conn:
            table = self._table
            query = select(table).where(table.c[self._idcol] == self._id)
            self.log.debug(f"{query}")
            result = conn.execute(query).fetchone()
            self._data = to_data(result)

    def __repr__(self):
        return f"{self.name}:{self.id}"


# ---------------------------------------------------------------------------
# Attribute Hierarchy
#       Area  Hierarchy
#       Skill Hierarchy
# ---------------------------------------------------------------------------

class AttributeDetail(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
        self.check_data()
        self.parent = None
        self.children = []

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def name(self) -> str:
        return self.data['attribute']

    @property
    def description(self) -> str:
        return self.data['description']

    @property
    def parent_id(self) -> Optional[int]:
        """For the root, 'parent_id' is None"""
        return self.data['parent_id']

    @property
    def hierarchy_id(self):
        return self.data['attribute_master_id']
# end


class AttributeHierarchy(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)

    @property
    def name(self) -> str:
        self.check_data()
        return self.data['attribute_master_name']

    @property
    def description(self) -> str:
        self.check_data()
        return self.data['attribute_desc']

    @property
    def type(self) -> Literal["area", "skill"]:
        self.check_data()
        type = self.data['hierarchy_type']
        if type == 1:
            return "area"
        elif type == 2:
            return "skill"
        else:
            raise ValueError(f"Unsipported hierarchy type {type}")

    def details(self) -> list[AttributeDetail]:
        # table = self.ipom.AttributeDetail
        # idlist = self.node_ids()
        # return [AttributeDetail(self.ipom, id, table) for id in idlist]
        with self.engine.connect() as conn:
            tdetail = self.ipom.AttributeDetail
            query = select(tdetail).where(tdetail.c['attribute_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        return [AttributeDetail(self.ipom, to_data(result), tdetail) for result in rlist]

    def features(self) -> list[AttributeDetail]:
        return self.details()

    def feature_ids(self, leaf_only=False, with_name=False) -> Union[list[int], dict[int, str]]:
        """
        Return the id's list of all hierarchy's nodes
        If it is specified 'with_name=True', it is returned the dictionary {id: name}

        :param leaf_only: if True, only the leaf nodes are returned
        :param with_name: if to return the dictionary {id: name}
        :return:
        """
        with self.engine.connect() as conn:
            table = self.ipom.AttributeDetail
            if leaf_only:
                query = select(table.c['id', 'attribute']).where(
                    (table.c['attribute_master_id'] == self.id) &
                    (table.c['is_leafattribute'] == True)       # WARN: DOESN'T change '== True' into 'is True'
                )
            else:
                query = select(table.c['id', 'attribute']).where(
                    (table.c['attribute_master_id'] == self.id)
                )

            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # rlist: [(id, name), ...]
            if with_name:
                return {rec[0]: rec[1] for rec in rlist}
            else:
                return [rec[0] for rec in rlist]

    def tree(self) -> AttributeDetail:
        root = None
        nodes = self.details()
        node_dict = {}
        for node in nodes:
            node_dict[node.id] = node

        for node in nodes:
            parent_id = node.parent_id
            if parent_id is None:
                root = node
                continue
            parent = node_dict[parent_id]
            node.parent = parent
            parent.children.append(node)
        return root

    def to_ids(self, attr: Union[None, int, list[int], str, list[str]], leaf_only=False) -> list[int]:
        """
        Convert the attribute(s) in a list of attribute ids.
        The attribute can be specified as id (an integer) or name (a string)
        If attr is None, all leaf attributes are returned

        :param attr: attribute(s) to convert
        :param leaf_only: if to select the leaf attributes
        :return: list of attribute ids
        """
        feature_dict = self.feature_ids(with_name=True)
        feature_drev = reverse_dict(feature_dict)
        aids = []

        if attr is None:
            return self.feature_ids(leaf_only=leaf_only)

        if isinstance(attr, int | str):
            attr = [attr]
        if is_instance(attr, list[int]):
            for aid in attr:
                if aid not in feature_dict:
                    raise ValueError(f"Invalid attribute {aid}: not available in hierarchy {self.name}")
                else:
                    aids.append(aid)
            return attr
        elif is_instance(attr, list[str]):
            for aname in attr:
                if not aname in feature_drev:
                    raise ValueError(f"Invalid attribute {aname}: not available in hierarchy {self.name}")
                aids.append(feature_drev[aname])
        else:
            raise ValueError(f"Invalid attribute type {type(attr)}: unsupported in in hierarchy {self.name}")

        return aids
    # end

    def delete(self):
        self.ipom.delete_attribute_hierarchy(self.id, self.type)
# end


class PeriodHierarchy(IPlanObject):

    SUPPORTED_FREQS = {
        'day': 'D',     # day start
        'week': 'W',    # week start
        'month': 'M'    # month start
    }

    def __init__(self, ipom, period_hierarchy, period_length):
        super().__init__(ipom)
        assert period_hierarchy in self.SUPPORTED_FREQS

        # period_hierarchy
        self._period_hierarchy = period_hierarchy
        self._period_length = period_length

    @property
    def freq(self) -> Literal['D', 'W', 'M']:
        """Frequency Pandas's compatible"""
        return self.SUPPORTED_FREQS[self._period_hierarchy]

    @property
    def periods(self) -> int:
        return self._period_length

    def date_range(self, start=None, end=None, periods=None) -> pd.DatetimeIndex:
        assert is_instance(start, Union[None, datetime])
        assert is_instance(end, Union[None, datetime])
        assert is_instance(periods, Union[None, int])
        return pd.date_range(start=start, end=end, periods=periods, freq=self.freq)

    def period_range(self, start=None, end=None, periods=None) -> pd.PeriodIndex:
        assert is_instance(start, Union[None, datetime])
        assert is_instance(end, Union[None, datetime])
        assert is_instance(periods, Union[None, int])
        return pd.period_range(start=start, end=end, periods=periods, freq=self.freq)

    def __repr__(self):
        return f"{self._period_hierarchy}:{self._period_length}"
# end


class AttributeHierarchies(IPlanObject):

    def __init__(self, ipom):
        super().__init__(ipom)

    def area_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        return self.ipom.attribute_hierarchy(id, "area")

    def skill_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        return self.ipom.attribute_hierarchy(id, "skill")

    def create_area_hierarchy(self, name: str, hierarchy_tree) -> AttributeHierarchy:
        return self.ipom.create_attribute_hierarchy(name, hierarchy_tree, 'area')

    def create_skill_hierarchy(self, name: str, hierarchy_tree) -> AttributeHierarchy:
        return self.ipom.create_attribute_hierarchy(name, hierarchy_tree, 'skill')
# end


# ---------------------------------------------------------------------------
# IDataModelMaster  (Data Model)
# IDataModelDetail  (Measure)
# IDataMaster       (Data Master)
# ---------------------------------------------------------------------------

class Measure(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
        self.check_data()

    @property
    def name(self) -> str:
        return self.data['measure_id']

    @property
    def description(self) -> str:
        return self.data['description']

    # leaf_formula
    # non_leaf_formula

    @property
    def type(self, leaf=True) -> Literal["INPUT", "FEED", "CALCULATION"]:
        return self.data['type'] if leaf else self.data['non_leaf_type']

    @property
    def data_model(self):
        data_model_id = self.data['data_model_id_fk']
        return self.ipom.data_model(data_model_id)

    # skills ???
    # skill_enabled
    # popup_id ???

    # default_value
    # positive_only
    # model_precision

    @property
    def mode(self) -> Literal["PLAN", "SCENARIO"]:
        return self.data['measure_mode']

    # linked_measure ???
    # period_agg_type ???
# end


class IDataModel(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
        self.check_data()

    @property
    def name(self) -> str:
        return self.data['description']

    @property
    def description(self) -> str:
        return self.data['description']

    def details(self) -> list[Measure]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataModelDetail
            query = select(table).where(table.c['data_model_id_fk'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        return [Measure(self.ipom, to_data(result), table) for result in rlist]

    def measures(self) -> list[Measure]:
        return self.details()

    def measure(self, id: Union[int, str]):
        with self.engine.connect() as conn:
            table = self.ipom.iDataModelDetail
            if isinstance(id, str):
                query = select(table).where(
                    (table.c['measure_id'] == id) &
                    (table.c['data_model_id_fk'] == self.id)
                )
                self.log.debug(query)
                ret = conn.execute(query).fetchone()
            else:
                query = select(table).where(
                    (table.c.id == id) &
                    (table.c['data_model_id_fk'] == self.id)
                )
                self.log.debug(query)
                ret = conn.execute(query).fetchone()
            return Measure(self.ipom, to_data(ret), table)
# end


class IDataMaster(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
        self.check_data()

    @property
    def name(self) -> str:
        return self.data['description']

    @property
    def description(self) -> str:
        return self.data['description']

    @property
    def data_model(self) -> IDataModel:
        data_model_id = self.data['idatamodel_id_fk']
        return self.ipom.data_model(data_model_id)

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        area_hierarchy_id = self.data['area_id_fk']
        return self.ipom.area_hierarchy(area_hierarchy_id)

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        skill_hierarchy_id = self.data['skill_id_fk']
        return self.ipom.skill_hierarchy(skill_hierarchy_id)

    @property
    def period_hierarchy(self) -> PeriodHierarchy:
        period_hierarchy = self.data['period_hierarchy']
        period_length = self.data['period']
        return PeriodHierarchy(self.ipom, period_hierarchy, period_length)
# end


# ---------------------------------------------------------------------------
# IDataValuesMaster == IPredictionPlans
# IPredictionPlan
# ---------------------------------------------------------------------------

class IPredictionPlans(IPlanObject):
    """
    Object used to retrieve the prediction interval based on the "Prediction Plan"
    A "Prediction Plan" has a name and it specified an interval for each area in
    area hierarchy.

    The prediction plan can be selected:

        - by id
        - by name
        - by date contained in the prediction interval

    To generalize, the area is not mandatory and the name can be a partial name: in this case
    it is possible to select multiple prediction plans
    If multiple prediction plans are selected, the interval to consider will be composed
    by

        - the minimum 'start_date'
        - the maximum 'end_date'

    Note: (None, None) is the 'empty interval'
    """

    def __init__(self, ipom: "IPlanObjectModel"):
        super().__init__(ipom)

    # -----------------------------------------------------------------------

    def exists_plan(self, name_or_date: Union[str, datetime], data_master_id: int) -> bool:
        assert is_instance(name_or_date, Union[str, datetime])

        if isinstance(name_or_date, str):
            name: str = name_or_date

            with self.engine.connect() as conn:
                table = self.ipom.iDataValuesMaster
                query = select(func.count()).select_from(table).where(
                    table.c.name.like(f"%{name}%") &
                    (table.c['idata_master_fk'] == data_master_id)
                )
                self.log.debug(query)
                count = conn.execute(query).scalar()
        elif is_instance(name_or_date, datetime):
            date: datetime = name_or_date

            with self.engine.connect() as conn:
                table = self.ipom.iDataValuesMaster
                query = select(func.count()).select_from(table).where(
                    (table.c['start_date'] <= date) &
                    (table.c['end_date'] >= date) &
                    (table.c['idata_master_fk'] == data_master_id)
                )
                self.log.debug(query)
                count = conn.execute(query).scalar()
        else:
            raise ValueError(f"Unsupported type for {name_or_date}")

        return count > 0

    def delete_plan(self, name: str):
        assert is_instance(name, str)

        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            query = delete(table).where(
                table.c.name.like(f"%{name}%")
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()

    def create_plan(self,
                    name: Optional[str],
                    data_master: Union[int, str, IDataMaster],
                    start_date: Union[None, datetime, tuple[datetime, datetime]],
                    end_date: Optional[datetime] = None,
                    area_feature_ids: Union[None, int, list[int]] = None,
                    force=False):
        assert is_instance(name, Optional[str])
        assert is_instance(data_master, Union[int, str, IDataMaster])
        assert is_instance(start_date, Union[datetime, tuple[datetime, datetime]])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(area_feature_ids, Union[None, int, list[int]])

        #
        # prepare the data
        #
        now: datetime = datetime.now()
        area_feature_ids = as_list(area_feature_ids)
        """:type: list[int]"""

        # retrieve the Data Master
        if isinstance(data_master, int | str):
            data_master = self.ipom.data_master(data_master)
        data_master_id = data_master.id

        if name is None:
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            name = f"Auto_Plan_OM_{now_str}"

        already_exists = self.exists_plan(name, data_master_id)
        if already_exists and not force:
            self.log.warning(f"Plan {name} already existent")
            return self
        if already_exists:
            self.log.warning(f"Delete plan {name}")
            self.delete_plan(name)

        af_ids: list[int] = data_master.area_hierarchy.feature_ids()
        if len(area_feature_ids) == 0:
            area_feature_ids = af_ids
        else:
            af_count = len(area_feature_ids)
            ai_count = len(set(af_ids).intersection(area_feature_ids))
            if af_count != ai_count:
                self.log.error("Found incompatible area_feature_ids")
                self.log.error(f"      data_master id: {data_master_id}")
                self.log.error(f"   master's area ids: {list(af_ids)}")
                self.log.error(f"    area_feature_ids: {area_feature_ids}")
        # end

        #
        # parse (start_date, end_date)
        #

        # if end_date is not specified, it is computed as 'start_date' + period_length
        if isinstance(start_date, tuple | list):
            start_date, end_date = start_date

        # compute end_date based on start_date & period_length
        if end_date is None:
            periods = data_master.period_hierarchy.periods
            end_date = start_date + timedelta(days=periods)

        #
        # create the plans for each area
        #

        # [tb_idata_values_master]
        # -- id
        #  1) start_date
        #  2) end_date
        #  3) name
        #  4) created_date
        #  5) idata_master_fk
        #  6) loan_updated_time
        #  7) published
        #  8) isscenario
        #  9) temp_ind
        # 10) area_id
        # 11) last_updated_date
        # 12) published_id
        # 13) note

        # STUPID implementation
        count = 0
        with (self.engine.connect() as conn):
            table = self.ipom.iDataValuesMaster
            for area_feature_id in area_feature_ids:
                stmt = insert(table).values(
                    start_date=start_date,
                    end_date=end_date,
                    name=name,
                    created_date=now,
                    idata_master_fk=data_master_id,
                    loan_updated_time=now,
                    published='N',
                    isscenario='N',
                    temp_ind='N',
                    area_id=area_feature_id,
                    last_updated_date=None,
                    published_id=None,
                    note="created by " + CREATED_BY
                ).returning(table.c.id)
                if count == 0: self.log.debug(stmt)
                rec_id = conn.execute(stmt).scalar()
                count += 1
            conn.commit()
        return
    # end

    # -----------------------------------------------------------------------

    def select_date_interval(self, id_or_name_or_date: Union[None, int, str, datetime] = None,
                             data_master_id: int = 0,
                             area_feature_ids: Union[None, int, list[int]] = None) \
        -> Optional[tuple[datetime, datetime]]:
        """
        Retrieve the date interval used for the prediction based on several rules:

            - prediction plan id
            - prediction plan name
            - date contained in the date interval
            - data_master_id

        The prediction plan must be specific for a selected Data Master
        It is possible to specify the area(s). If the areas are not specified, it is
        selected the date interval as min(start_date), max(end_date) for all defined
        areas. If no plan is found, it is returned None

        :param id_or_name_or_date: Prediction Plan id or name or date contained in the prediction interval
        :data_master_id: id of the Data Master to use
        :param area_feature_ids: specific area(s) to consider
        :return: (start_date, end_date) OR None if no interval is found
        """
        assert is_instance(id_or_name_or_date, Union[None, int, str, datetime])
        assert is_instance(data_master_id, int)
        assert is_instance(area_feature_ids, Union[None, int, list[int]])

        area_feature_ids: list[int] = as_list(area_feature_ids)
        # convert a string representing an integer value into a integer
        id_or_name_or_date = safe_int(id_or_name_or_date)

        if id_or_name_or_date is None:
            return self._select_by_data_master(data_master_id, area_feature_ids)
        elif isinstance(id_or_name_or_date, datetime):
            start_end_date = self._select_by_date(id_or_name_or_date, data_master_id, area_feature_ids)
        elif isinstance(id_or_name_or_date, int):
            start_end_date = self._select_by_id(id_or_name_or_date, data_master_id, area_feature_ids)
        elif isinstance(id_or_name_or_date, str):
            start_end_date = self._select_by_name(id_or_name_or_date, data_master_id, area_feature_ids)
        else:
            raise ValueError(f"Unsupported type for value {id_or_name_or_date}")

        # (None, None) -> None
        if start_end_date[0] is None or start_end_date[1] is None:
            return None
        else:
            return start_end_date

    def _select_by_data_master(self, data_master_id: int, area_feature_ids: list[int]) -> tuple[datetime, datetime]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            if len(area_feature_ids) == 0:
                query = select(table.c['start_date', 'end_date']).where(
                    (table.c['idata_master_fk'] == data_master_id)
                )
            else:
                query = select(table.c['start_date', 'end_date']).where(
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c['area_id'].in_(area_feature_ids))
                )
            self.log.debug(query)
            result = conn.execute(query).fetchone()
            return result[0], result[1]

    def _select_by_id(self, ppid: int, data_master_id: int, area_feature_ids: list[int]) -> tuple[datetime, datetime]:
        # data_master_id & area_feature_ids are not necessary BUT they are used to force consistency
        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            if len(area_feature_ids) == 0:
                query = select(table.c['start_date', 'end_date']).where(
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c.id == ppid)
                )
            else:
                query = select(table.c['start_date', 'end_date']).where(
                    (table.c.id == ppid) &
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c['area_id'].in_(area_feature_ids))
                )
            self.log.debug(query)
            result = conn.execute(query).fetchone()
            return result[0], result[1]

    def _select_by_name(self, name: str, data_master_id: int, area_feature_ids: list[int]) -> tuple[datetime, datetime]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            if len(area_feature_ids) == 0:
                query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c['name'].like(f"%{name}%"))
                )
            else:
                query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c['name'].like(f"%{name}%")) &
                    (table.c['area_id'].in_(area_feature_ids))
                )
            self.log.debug(query)
            result = conn.execute(query).fetchone()
            return (None, None) if result is None else result[0], result[1]

    def _select_by_date(self, when: datetime, data_master_id: int, area_feature_ids: list[int]) -> tuple[
        datetime, datetime]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            if len(area_feature_ids) == 0:
                query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c['start_date'] <= when) &
                    (table.c['end_date'] >= when)
                )
            else:
                query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                    (table.c['idata_master_fk'] == data_master_id) &
                    (table.c['start_date'] <= when) &
                    (table.c['end_date'] >= when) &
                    (table.c['area_id'].in_(area_feature_ids))
                )
            self.log.debug(query)
            result = conn.execute(query).fetchone()
            return result[0], result[1]
# end


class IPredictionPlan(IPlanObject):

    def __init__(self, ipom, name: str, data_master: Union[int, str]):
        super().__init__(ipom)
        assert is_instance(name, str)
        assert is_instance(data_master, Union[int, str])
        self._name: str = name
        self._data_master = self.ipom.data_master(data_master)
    # end

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_master(self) -> IDataMaster:
        return self._data_master

    @property
    def plan_ids(self) -> list[int]:
        return self.ipom.select_plan_ids(
            self.name,
            [self.data_master.id],
            self.data_master.area_hierarchy.feature_ids(leaf_only=True)
        )

    def area_plan_map(self) -> dict[int, int]:
        """
        Map area_feature_id -> plan_id
        """
        name = self._name
        data_master_id = self._data_master.id

        with self.ipom.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            query = select(table.c['area_id', 'id']).where(
                (table.c['name'] == name) &
                (table.c['idata_master_fk'] == data_master_id)
            )
            self.log.debug(query)
            pmap = {}
            rlist = conn.execute(query)
            for r in rlist:
                # area_feature_id -> plan_id
                pmap[r[0]] = r[1]

        return pmap

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        return self.data_master.area_hierarchy

    def exists(self) -> bool:
        name = self._name
        data_master_id = self._data_master.id

        with self.ipom.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            query = select(func.count()).select_from(table).where(
                (table.c.name.like(f"%{name}%")) &
                (table.c['idata_master_fk'] == data_master_id)
            )
            self.log.debug(query)
            count = conn.execute(query).scalar()
        return count > 0

    def delete(self):
        name = self._name
        data_master_id = self._data_master.id

        with self.ipom.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            query = delete(table).where(
                (table.c.name.like(f"%{name}%")) &
                (table.c['idata_master_fk'] == data_master_id)
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()

    def create(self, start_date: datetime, end_date: Optional[datetime] = None,
               periods: Optional[int] = None, note: Optional[str] = None,
               update: Optional[bool] = None) -> Self:
        """
        Create a plan with the specified name for all areas in the area hierarchy
        If end_date or periods are not specified, it is used the PeriodHierarchy of the DataMaster

        :param start_date: start date
        :param end_date: end date
        :param periods: n of periods
        :param update: how to update the data already present in the database
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        :return:
        """

        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])

        name = self._name

        #
        # delete the plan if 'force == True'
        #
        already_exists = self.exists()
        if already_exists and not update:
            self.log.warning(f"Plan {name} already existent")
            return self

        if already_exists:
            self.log.warning(f"Delete plan {name}")
            self.delete()

        #
        # parse 'end_date'
        #
        if end_date is None:
            periods = self.data_master.period_hierarchy.periods
            freq = self.data_master.period_hierarchy.freq
            end_date = start_date + relativeperiods(periods, freq)

        #
        # create the plans for each area
        #

        # [tb_idata_values_master]
        # -- id
        #  1) start_date
        #  2) end_date
        #  3) name
        #  4) created_date
        #  5) idata_master_fk
        #  6) loan_updated_time
        #  7) published
        #  8) isscenario
        #  9) temp_ind
        # 10) area_id
        # 11) last_updated_date
        # 12) published_id
        # 13) note
        self.log.info(f"Create plan {name}")

        # STUPID implementation
        data_master_id = self.data_master.id
        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        now: datetime = datetime.now()
        note = "created by " + CREATED_BY if note is None else note
        count = 0
        with (self.engine.connect() as conn):
            table = self.ipom.iDataValuesMaster
            for area_feature_id in area_feature_dict:
                area_name = area_feature_dict[area_feature_id]
                self.log.debugt(f"... create plan for {area_name}")

                stmt = insert(table).values(
                    start_date=start_date,
                    end_date=end_date,
                    name=name,
                    created_date=now,
                    idata_master_fk=data_master_id,
                    loan_updated_time=now,
                    published='N',
                    isscenario='N',
                    temp_ind='N',
                    area_id=area_feature_id,
                    last_updated_date=None,
                    published_id=None,
                    note=note
                ).returning(table.c.id)
                if count == 0: self.log.debug(stmt)
                rec_id = conn.execute(stmt).scalar()
                count += 1
            conn.commit()
        # end
        self.log.info(f"Done")
        return self
    # end

    def __repr__(self):
        return f"{self.name}[{self._data_master.name}]"
# end


# ---------------------------------------------------------------------------
# IPredictDetailFocussed
# ---------------------------------------------------------------------------

class IPredictDetailFocussed(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)

    @property
    def name(self) -> str:
        self.check_data()
        return self.data['parameter_desc']

    @property
    def description(self) -> str:
        self.check_data()
        return self.data['parameter_desc']

    @property
    def is_target(self) -> bool:
        return self.type == 'output'

    @property
    def type(self) -> Literal["input", "ouput"]:
        self.check_data()
        return self.data['parameter_value']

    @property
    def measure(self, write: bool = False) -> Optional[Measure]:
        """
        Retrieve the measure containing the data or the measure used to save
        the predicted data

        :param write: is to select the measure used to save the predicted data
        :return: a measure
        """

        self.check_data()
        measure_id = self.data['to_populate'] if write else self.data['parameter_id']
        tmeasure = self.ipom.iDataModelDetail
        return Measure(self.ipom, measure_id, tmeasure)
# end


class IPredictMasterFocussed(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
        self.check_data()

        # local caches
        self._data_master = None
        self._data_model = None
        self._area_hierarchy = None
        self._skill_hierarchy = None
        pass

    @property
    def name(self) -> str:
        return self.data['ipr_conf_master_name']

    @property
    def description(self) -> str:
        return self.data['ipr_conf_master_desc']

    # @property
    # def data_master(self) -> IDataMaster:
    #     if self._data_master is not None:
    #         return self._data_master
    #
    #     # check 'idata_id_fk'
    #     data_master_id = self.data['idata_id_fk']
    #     if data_master_id is not None:
    #         self._data_master = self.ipom.data_master(data_master_id)
    #     else:
    #         data_model_id = self.data['idata_model_details_id_fk']
    #         area_hierarchy_id = self.data['area_id_fk']
    #         skill_hierarchy_id = self.data['skill_id_fk']
    #         self._data_master = self.ipom.find_data_master(data_model_id, area_hierarchy_id, skill_hierarchy_id)
    #
    #     return self._data_master

    @property
    def data_model(self) -> IDataModel:
        if self._data_model is None:
            data_model_id = self.data['idata_model_details_id_fk']
            self._data_model = self.ipom.data_model(data_model_id)
        return self._data_model

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        if self._area_hierarchy is None:
            area_hierarchy_id = self.data['area_id_fk']
            self._area_hierarchy = self.ipom.area_hierarchy(area_hierarchy_id)
        return self._area_hierarchy

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        if self._skill_hierarchy is None:
            skill_hierarchy_id = self.data['skill_id_fk']
            self._skill_hierarchy = self.ipom.skill_hierarchy(skill_hierarchy_id)
        return self._skill_hierarchy

    @property
    def parameters(self) -> list[IPredictDetailFocussed]:
        return self.details()

    @property
    def input_target_parameters(self) -> tuple[list[IPredictDetailFocussed], list[IPredictDetailFocussed]]:
        inputs: list[IPredictDetailFocussed] = []
        targets: list[IPredictDetailFocussed] = []

        parameters = self.details()
        for param in parameters:
            if param.type == 'output':
                targets.append(param)
            elif param.type == 'input':
                inputs.append(param)
            else:
                raise ValueError(f"Unsupported parameter {param}")

        return inputs, targets

    @property
    def input_target_measures(self) -> tuple[list[Measure], list[Measure]]:
        input_measures = []
        target_measures = []
        for param in self.parameters:
            if param.is_target:
                target_measures.append(param.measure)
            else:
                input_measures.append(param.measure)
        return input_measures, target_measures

    @property
    def measures(self) -> list[Measure]:
        input_measures, target_measures = self.input_target_measures
        return input_measures + target_measures

    @property
    def input_target_measure_ids(self) -> tuple[list[int], list[int]]:
        input_ids = []
        target_ids = []
        with self.engine.connect() as conn:
            tdetail = self.ipom.iPredictDetailFocussed
            query = select(tdetail.c['parameter_id', 'parameter_value']).where(
                tdetail.c['ipr_conf_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            for result in rlist:
                id = result[0]
                type = result[1]
                if type == 'output':
                    target_ids.append(id)
                elif type == 'input':
                    input_ids.append(id)
                else:
                    raise ValueError(f"Unsupported parameter type {type}")
        return input_ids, target_ids

    # end

    # @property
    # def measure_ids(self) -> list[int]:
    #     input_ids, target_ids = self.input_target_measure_ids
    #     return input_ids + target_ids

    def measures_ids(self, with_name=False) -> Union[list[int], dict[int, str]]:
        input_ids, target_ids = self.input_target_measure_ids
        measure_ids = input_ids + target_ids
        if with_name:
            return self.ipom.select_measure_names(measure_ids)
        else:
            return measure_ids

    # Note: the COLUMN 'tb_ipr_conf_master_focussed.idata_id_fk' IS NULL
    #       in ALL records in the table!
    #       HOWEVER, in "theory" it is possible to retrieve teh 'Data Master' based on the
    #       values:  (data_model_id, area_hierarchy_id, skill_hierarchy_id)

    def details(self) -> list[IPredictDetailFocussed]:
        with self.engine.connect() as conn:
            tdetail = self.ipom.iPredictDetailFocussed
            query = select(tdetail).where(tdetail.c['ipr_conf_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        return [IPredictDetailFocussed(self.ipom, to_data(result), tdetail) for result in rlist]
# end


# ---------------------------------------------------------------------------
# IDataValuesMaster
# IDataValuesDetail
# IDataValuesDetailHist
# ---------------------------------------------------------------------------

class IDataValuesMaster(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
        self.check_data()

    @property
    def name(self) -> str:
        return self.data['name']

    @property
    def data_master(self) -> IDataMaster:
        data_master_id = self.data['idata_master_fk']
        return self.ipom.data_master(data_master_id)

    # @property
    # def data_model(self):
    #     return self.data_master.data_model

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        # check if the 'area_id' is consistent with the Area Hierarchy defined in Data Master
        area_hierarchy_id = self.ipom.area_feature(self.data['area_id']).hierarchy_id
        area_hierarchy = self.data_master.area_hierarchy
        assert area_hierarchy_id == area_hierarchy.id
        return area_hierarchy

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        return self.data_master.skill_hierarchy

    @property
    def start_date(self) -> datetime:
        return self.data['start_date']

    @property
    def end_date(self) -> datetime:
        return self.data['end_date']

    @property
    def area_feature(self) -> AttributeDetail:
        area_feature_id = self.data['area_id']
        return self.ipom.area_feature(area_feature_id)

    def select_data_values(self):
        return self.ipom.select_data_values(self.id)
# end


class IDataValue(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)
# end


# ---------------------------------------------------------------------------
# IPredictTimeSeries
# ---------------------------------------------------------------------------

class IPredictTimeSeries(IPlanObject):

    def __init__(self, ipom, id: int, data_master_id: int):
        super().__init__(ipom)

        assert is_instance(id, int)
        assert is_instance(data_master_id, int)

        self._id: int = id
        self._plan: Optional[str] = None

        self._data_master: IDataMaster = self.ipom.data_master(data_master_id)
        self._pf: IPredictMasterFocussed = self.ipom.predict_master_focussed(id)
    # end

    # -----------------------------------------------------------------------
    # Delegate to PredictFocussed
    # -----------------------------------------------------------------------

    @property
    def id(self):
        return self._pf.id

    @property
    def name(self):
        return self._pf.name

    @property
    def data_master(self) -> IDataMaster:
        return self._data_master

    @property
    def data_model(self) -> IDataModel:
        return self._data_master.data_model

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        # same than self.data_model.area_hierarchy
        return self._pf.area_hierarchy

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        # same than self.data_model.skill_hierarchy
        return self._pf.skill_hierarchy

    @property
    def period_hierarchy(self) -> PeriodHierarchy:
        return self.data_master.period_hierarchy

    @property
    def input_target_measure_ids(self) -> tuple[list[int], list[int]]:
        return self._pf.input_target_measure_ids

    def measure_ids(self, with_name=False) -> Union[list[int], dict[int, str]]:
        return self._pf.measures_ids(with_name=with_name)

    # -----------------------------------------------------------------------
    # Prediction plan
    # -----------------------------------------------------------------------

    @property
    def plan(self) -> Optional[IPredictionPlan]:
        return self._plan

    def set_plan(self, plan: str):
        assert is_instance(plan, str)
        self._plan = plan
        return self

    # -----------------------------------------------------------------------
    # Train/predict data
    # -----------------------------------------------------------------------

    def select_train_data(
        self,
        plan: Optional[str] = None,
        area: Union[None, int, list[int]] = None,
        skill: Union[None, int, list[int]] = None,
        new_format=True) -> DataFrame:
        """
        Retrieve the train data

        :param plan: name of the plan used for reference
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param new_format: DataFrame format
        :return: the dataframe used for training. It contains input/target features
        """

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])
        assert is_instance(plan, Optional[str])

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        df: DataFrame = self.ipom.select_train_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            new_format=new_format
        )
        return df
    # end

    def select_predict_data(
        self,
        plan: Optional[str] = None,
        area: Union[None, int, list[int], str, list[str]] = None,
        skill: Union[None, int, list[int], str, list[str]] = None,
        new_format=True) -> DataFrame:
        """
        Retrieve predict data

        :param plan:
        :param area:
        :param skill:
        :param new_format:
        :return:
        """
        assert is_instance(plan, Optional[str])
        assert is_instance(area, Union[None, int, list[int]])
        assert is_instance(skill, Union[None, int, list[int]])

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        df: DataFrame = self.ipom.select_predict_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            new_format=new_format
        )
        return df
    # end


    def select_predict_data_ext(
        self,
        plan: Optional[str] = None,
        area: Union[None, int, list[int], str, list[str]] = None,
        skill: Union[None, int, list[int], str, list[str]] = None,
        #
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        periods: Optional[int] = None,
        freq: Literal['D', 'W', 'M'] = 'D',
        #
        new_format=True) -> DataFrame:
        """
        Retrieve predict data

        :param plan: name of the plan used for reference
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param start_date: optional start date
        :param end_date: optional end date
        :param periods: n of periods. The period depends on the DataModel::PeriodHierarchy
        :param freq: period frequency
        :param new_format: DataFrame format
        :return: the dataframe used for prediction. It contains the input features only
        """
        assert is_instance(plan, Optional[str])
        assert is_instance(area, Union[None, int, list[int]])
        assert is_instance(skill, Union[None, int, list[int]])

        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])
        assert is_instance(freq, Literal['D', 'W', 'M'])

        area_feature_dict = self.area_hierarchy.feature_ids(leaf_only=False, with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(leaf_only=False, with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        input_feature_ids, _ = self.input_target_measure_ids
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        freq = self.period_hierarchy.freq if freq is None else freq
        periods = self.period_hierarchy.periods if periods is None else periods
        if start_date is not None and end_date is None:
            end_date = start_date + relativeperiods(periods=periods, freq=freq)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        df: DataFrame = self.ipom.select_predict_data_ext(
            data_master_id,
            plan_ids,
            area_feature_dict, skill_feature_dict,
            input_feature_ids, measure_dict,
            start_date, end_date, freq,
            new_format=new_format
        )
        return df
    # end

    def save_train_data(self, df: DataFrame, plan: Optional[str] = None, update: Optional[bool] = None):
        """
        Save the data for training (table: 'tb_idata_values_detail_hist').
        The dataframe can be passed in the following formats:

            - old format:
                columns ['area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id>, ...]
                index: not used
                Note: <measure_id> can be an integer or a string
                      'area_id_fk', 'skill_id_fk' are integer values (area/skill id)
            - old format/multiindex:
                columns ['area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id>, ...]
                index: 'area_id_fk'/'skill_id_fk'/'time'
                Note: <measure_id> can be an integer or a string
            - new format:
                columns ['area', 'skill', 'date', <measure_name>, ...]
                index: not used
                Note: 'area' and 'skill' values are strings
            - new format/multiindex:
                columns [<measure_name>, ...]
                index: 'area'/'skill'/'date'
                Note: 'area' and 'skill' values are strings

        The dataframe replace all data already present in the same measure

        :param df: dataframe to insert into database
        :param plan: name of the plan used for reference
        :param update: how to update the data already present in the database
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        """
        assert is_instance(df, DataFrame)
        assert is_instance(plan, Optional[str])

        self.log.info("Save train data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        area_plan_map = pplan.area_plan_map()

        # if plan is not None:
        #     pplan = self.ipom.prediction_plan(plan, data_master_id)
        #     area_plan_map = pplan.area_plan_map()
        # else:
        #     area_plan_map = None

        # 1) normalize the dataframe
        #    columns: ['area_id_fk', 'skill_id_fk', 'state_date', <measure_id1>, ...]
        df = normalize_df(df, area_feature_dict, skill_feature_dict, measure_dict)

        # 2) split df by area/skill (and drop the columns)
        dfdict = pdx.groups_split(df, groups=['area_id_fk', 'skill_id_fk'], drop=True)
        for area_skill in dfdict:
            area_feature_id: int = area_skill[0]
            skill_feature_id: int = area_skill[1]
            dfas = dfdict[area_skill]
            for measure_id in measure_dict:
                area_name = area_feature_dict[area_feature_id]
                skill_name = skill_feature_dict[skill_feature_id]
                measure_name = measure_dict[measure_id]
                plan_id = None if area_plan_map is None or area_feature_id not in area_plan_map \
                    else area_plan_map[area_feature_id]
                try:
                    self.log.debugt(f"... [train] saving area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    # Note: type(*_feature_id) is a numpy type! converted into Python int
                    self.ipom.save_area_skill_train_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id), int(plan_id),
                        dfas, update=update)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to save train data for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return

    def delete_train_data(self,
                          plan: Optional[str] = None,
                          area: Union[None, int, list[int], str, list[str]] = None,
                          skill: Union[None, int, list[int], str, list[str]] = None,):
        """

        :param plan:
        :param area:
        :param skill:
        :return:
        """

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])

        self.log.info("Deleting train data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        self.ipom.delete_train_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
        )
        self.log.info("Done")
        return
    # end

    def delete_predict_data(self,
                            plan: Optional[str] = None,
                            area: Union[None, int, list[int]] = None,
                            skill: Union[None, int, list[int]] = None,):
        """

        :param plan:
        :param area:
        :param skill:
        :return:
        """
        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])

        self.log.info("Deleting predict data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, self.data_master.id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        # data_master_id: int,
        # data_values_master_ids: list[int],
        # area_feature_dict: dict[int, str],
        # skill_feature_dict: dict[int, str],
        # measure_dict: dict[int, str],
        # start_date: Optional[datetime] = None,
        # end_date: Optional[datetime] = None,
        # freq: Literal['D', 'W', 'M'] = 'D',
        # new_format=False):

        self.ipom.delete_predict_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
        )
        self.log.info("Done")
        return
    # end

    def save_predict_data(self, df: DataFrame,
                          plan: Optional[str] = None,
                          update: Optional[bool] = None):
        """
        Save the data for the prediction (table: 'tb_idata_values_detail').
        The dataframe can be passed in the following formats:

            - old format:
                columns ['area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id>, ...]
                index: not used
                Note: <measure_id> can be an integer or a string
                      'area_id_fk', 'skill_id_fk' are integer values (area/skill id)
            - old_format/multiindex:
                columns ['area_id_fk', 'skill_id_fk', 'time', 'day', <measure_id>, ...]
                index: 'area_id_fk'/'skill_id_fk'/'time'
                Note: <measure_id> can be an integer or a string
            - new format:
                columns ['area', 'skill', 'date', <measure_name>, ...]
                index: not used
                Note: 'area' and 'skill' values are strings
            - new format/multiindex:
                columns [<measure_name>, ...]
                index: 'area'/'skill'/'date'
                Note: 'area' and 'skill' values are strings

        The dataframe replace all data already present in the same measure

        Note: the dataframe must contain ALL measures, not only the 'input features'.
            This because it is necessary to save also the PAST target data.
            The targets to predict MUST be set as 'Not a Number'

        Note/2: how to handle the NOT predicted target data?
            There are some approaches:

                - it is used 0
                - it is used NaN

            Now, the problem is: the not predicted data is inserted in the database or not?

        :param df: dataframe to insert into database
        :param plan: name of the plan used for reference
        :param update: how to update the data already present in the database
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        """
        assert is_instance(df, DataFrame)
        assert is_instance(plan, Optional[str])

        self.log.info("Save predict data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        # 0) retrieve the Plan map
        #
        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        area_plan_map = pplan.area_plan_map()

        # 1) normalize the dataframe
        #    columns: ['area_id_fk', 'skill_id_fk', 'state_date', <measure_id1>, ...]
        df = normalize_df(df, area_feature_dict, skill_feature_dict, measure_dict)

        # 2) split df by area/skill (and drop the columns)
        dfdict = pdx.groups_split(df, groups=['area_id_fk', 'skill_id_fk'], drop=True)
        for area_skill in dfdict:
            area_feature_id: int = area_skill[0]
            skill_feature_id: int = area_skill[1]
            dfas = dfdict[area_skill]
            for measure_id in measure_dict:
                area_name = area_feature_dict[area_feature_id]
                skill_name = skill_feature_dict[skill_feature_id]
                measure_name = measure_dict[measure_id]
                plan_id = area_plan_map[area_feature_id]
                try:
                    self.log.debugt(f"... [pred] saving area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    # Note: type(*_feature_id) is a numpy type! converted into Python int
                    self.ipom.save_area_skill_predict_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id), int(plan_id),
                        dfas, update=update)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to save predict data for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return
# end


# ---------------------------------------------------------------------------
# IPlanObjectModel
# ---------------------------------------------------------------------------

class IPlanObjectModel(IPlanObject):

    def __init__(self, url: Union[str, dict, URL], **kwargs):
        assert is_instance(url, Union[str, dict, URL])
        if is_instance(url, dict):
            datasource_dict: dict = url
            url = URL.create(**datasource_dict)

        self.engine: Optional[Engine] = None
        super().__init__(self)
        self.url = url
        self.metadata: Optional[MetaData] = None
        self.kwargs = kwargs

    # -----------------------------------------------------------------------

    def connect(self, **kwargs) -> "IPlanObjectModel":
        """
        It creates a 'connection' to the DBMS, NOT a connection to execute queries (a 'session')
        This step is used to create the data structures necessary to access the specific database
        (Python SQLAlchemy engine, metadata, ...)
        :param kwargs: passed to SQLAlchemy 'create_engine(...)' function
        """
        self.log.debug(f"connecting to {self.url}")

        self.engine = create_engine(self.url, **kwargs)
        self._load_metadata()

        self.log.info(f"connected to {self.url}")
        return self

    def disconnect(self):
        """
        Release all resources used during the access to the DBMS/database
        """

        if self.engine is None:
            return
        self.engine.dispose(True)
        self.engine = None
        self.metadata = None
    # end

    # -----------------------------------------------------------------------
    # Support for
    #   with pom.connect():
    #       ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        pass

    # -----------------------------------------------------------------------
    # Area/Skill hierarchy

    def hierachies(self) -> AttributeHierarchies:
        return AttributeHierarchies(self)

    # --

    # def delete_area_hierarchy(self, id: Union[int, str]):
    #     self.delete_attribute_hierarchy(id, 'area')

    # def delete_skill_hierarchy(self, id: Union[int, str]):
    #     self.delete_attribute_hierarchy(id, 'skill')

    def delete_attribute_hierarchy(self, id: Union[int, str], attribute_type: Literal['area', 'skill']):
        assert is_instance(attribute_type, Literal['area', 'skill'])
        area_hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'],
                                             nullable=True)
        if area_hierarchy_id is None:
            return

        assert self.hierarchy_type(area_hierarchy_id) == attribute_type

        with self.engine.connect() as conn:
            # 1) delete tb_attribute_details
            table = self.AttributeDetail
            query = delete(table).where(table.c['attribute_master_id'] == area_hierarchy_id)
            self.log.debug(query)
            conn.execute(query)
            # 2) delete tb_attribute_master
            table = self.AttributeMaster
            query = delete(table).where(table.c.id == area_hierarchy_id)
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
    # end

    # --

    # def create_area_hierarchy(self, name: str, hierarchy_tree) \
    #         -> AttributeHierarchy:
    #     return self.create_attribute_hierarchy(name, hierarchy_tree, 'area')

    # def create_skill_hierarchy(self, name: str, hierarchy_tree) \
    #         -> AttributeHierarchy:
    #     return self.create_attribute_hierarchy(name, hierarchy_tree, 'skill')

    def create_attribute_hierarchy(self, name: str, hierarchy_tree, hierarchy_type: Literal['area', 'skill']) \
            -> AttributeHierarchy:
        assert is_instance(name, str)
        assert is_instance(hierarchy_type, Literal['area', 'skill'])

        if len(hierarchy_tree) == 1 and is_instance(hierarchy_tree, dict[str, list[str]]):
            return self._create_simple_hierarchy(name, hierarchy_tree, hierarchy_type)
        else:
            raise ValueError(f"Unsupported hierarchy tree format: {hierarchy_tree}")
    # end

    def _create_simple_hierarchy(self, name: str,
                                 hierarchy_tree: dict[str, list[str]],
                                 hierarchy_type: Literal['area', 'skill']) \
            -> AttributeHierarchy:
        now = datetime.now()
        root_name = list(hierarchy_tree.keys())[0]
        leaf_names = hierarchy_tree[root_name]
        description = name

        # hierarchy_tree:
        #   {parent: list[Union[str, dict[str, list]]}
        #   {child: parent}

        with self.engine.connect() as conn:
            # 1) create tb_attribute_master
            table = self.AttributeMaster
            query = insert(table).values(
                attribute_master_name=name,
                attribute_desc=description,
                createdby=CREATED_BY,
                createddate=now,
                hierarchy_type=1 if hierarchy_type == 'area' else 2
            ).returning(table.c.id)
            self.log.debug(query)
            hierarchy_id = conn.execute(query).scalar()
            # 2) create tb_attribute_detail
            #    simple format: {root: list[leaf]}
            table = self.AttributeDetail
            # 2.1) create the root
            query = insert(table).values(
                attribute_master_id=hierarchy_id,
                attribute=root_name,
                description=root_name,
                attribute_level=1,
                parent_id=None,
                createdby=CREATED_BY,
                createddate=now,
                is_leafattribute=False
            ).returning(table.c.id)
            self.log.debug(query)
            parent_id = conn.execute(query).scalar()
            # 2.2) create the leaves
            for leaf in leaf_names:
                query = insert(table).values(
                    attribute_master_id=hierarchy_id,
                    attribute=leaf,
                    description=leaf,
                    attribute_level=2,
                    parent_id=parent_id,
                    createdby=CREATED_BY,
                    createddate=now,
                    is_leafattribute=True
                ).returning(table.c.id)
                leaf_id = conn.execute(query).scalar()
            # end
            conn.commit()
        return AttributeHierarchy(self, hierarchy_id, self.AttributeMaster)
    # end

    # -----------------------------------------------------------------------

    def hierarchy_type(self, id: Union[int, str]) -> Literal["area", "skill"]:
        hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])

        with self.engine.connect() as conn:
            table = self.AttributeMaster
            query = select(table.c['hierarchy_type']).where(table.c['id'] == hierarchy_id)
            self.log.debug(f"{query}")
            hierarchy_type = conn.execute(query).fetchone()[0]
        return "area" if hierarchy_type == 1 else "skill"

    # def area_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
    #     area_hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
    #     ah = AttributeHierarchy(self.ipom, area_hierarchy_id, self.AttributeMaster)
    #     assert ah.type == "area"
    #     return ah

    @deprecated
    def area_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        return self.attribute_hierarchy(id, "area")

    @deprecated
    def area_feature(self, id: Union[int, str]) -> AttributeDetail:
        return self.attribute_detail(id, "area")

    # def area_feature(self, id: Union[int, str]) -> AttributeDetail:
    #     area_feature_id = self._convert_id(id, self.AttributeDetail, ['attribute', 'description'])
    #     af = AttributeDetail(self.ipom, area_feature_id, self.AttributeDetail)
    #     assert self.hierarchy_type(af.hierarchy_id) == "area"
    #     return af

    # def skill_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
    #     skill_hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
    #     sh = AttributeHierarchy(self.ipom, skill_hierarchy_id, self.AttributeMaster)
    #     assert sh.type == "skill"
    #     return sh

    @deprecated
    def skill_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        return self.attribute_hierarchy(id, "skill")

    @deprecated
    def skill_feature(self, id: Union[int, str]) -> AttributeDetail:
        return self.attribute_detail(id, "skill")

    # def skill_feature(self, id: Union[int, str]) -> AttributeDetail:
    #     skill_feature_id = self._convert_id(id, self.AttributeDetail, ['attribute', 'description'])
    #     sf = AttributeDetail(self.ipom, skill_feature_id, self.AttributeDetail)
    #     assert self.hierarchy_type(sf.hierarchy_id) == "skill"
    #     return sf

    def attribute_hierarchy(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']) \
            -> AttributeHierarchy:
        hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        hierarchy = AttributeHierarchy(self.ipom, hierarchy_id, self.AttributeMaster)
        assert hierarchy.type == hierarchy_type
        return hierarchy

    def attribute_detail(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']) \
            -> AttributeDetail:
        feature_id = self._convert_id(id, self.AttributeDetail, ['attribute', 'description'])
        detail = AttributeDetail(self.ipom, feature_id, self.AttributeDetail)
        assert self.hierarchy_type(detail.hierarchy_id) == hierarchy_type
        return detail

    # -----------------------------------------------------------------------
    # Data Model

    def data_model(self, id: Union[int, str]) -> IDataModel:
        data_model_id = self._convert_id(id, self.iDataModelMaster, ['description'])
        return IDataModel(self, data_model_id, self.iDataModelMaster)

    def delete_data_model(self, id: Union[int, str]):
        data_model_id = self._convert_id(id, self.iDataModelMaster, ['description'], nullable=True)
        if data_model_id is None:
            return

        with self.engine.connect() as conn:
            # 0) delete dependencies
            query = text("""
            DELETE FROM tb_ipr_conf_detail_focussed AS ticdf
            WHERE ticdf.parameter_id IN (
                SELECT timd.id FROM tb_idata_model_detail AS timd 
                WHERE timd.data_model_id_fk = :data_model_id
            )
            """)
            self.log.debug(query)
            conn.execute(query, parameters=dict(
                data_model_id=data_model_id
            ))

            query = text("""
            DELETE FROM tb_ipr_conf_master_focussed AS ticmf
            WHERE ticmf.idata_model_details_id_fk = :data_model_id
            """)
            table = self.iPredictMasterFocussed
            query = delete(table).where(
                table.c['idata_model_details_id_fk'] == data_model_id
            )
            self.log.debug(query)
            conn.execute(query, parameters=dict(
                data_model_id=data_model_id
            ))

            query = text("""
            DELETE FROM tb_idata_values_master AS tivm
            WHERE tivm.idata_master_fk IN (
                SELECT tim.id FROM tb_idata_master AS tim
                WHERE tim.idatamodel_id_fk = :data_model_id
            )
            """)
            self.log.debug(query)
            conn.execute(query, parameters=dict(
                data_model_id=data_model_id
            ))

            query = text("""
            DELETE FROM tb_idata_master AS tim
            WHERE tim.idatamodel_id_fk = :data_model_id
            """)
            self.log.debug(query)
            conn.execute(query, parameters=dict(
                data_model_id=data_model_id
            ))

            # 1) delete tb_data_model_detail
            table = self.iDataModelDetail
            query = delete(table).where(
                table.c['data_model_id_fk'] == data_model_id
            )
            self.log.debug(query)
            conn.execute(query)

            # 1) delete tb_data_model_master
            table = self.iDataModelMaster
            query = delete(table).where(
                table.c.id == data_model_id
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return
    # end

    def create_data_model(self, name: str, *,
                          targets: Union[str, list[str]],
                          inputs: Union[None, str, list[str]],
                          update: Optional[bool] = None) -> IDataModel:
        """

        :param name: Data Model name
        :param targets: measures used as FEED
        :param inputs: measures used as INPUT
        :param update: how to update the data already present in the database
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        :return:
        """
        assert is_instance(name, str)
        assert is_instance(targets, Union[str, list[str]])
        assert is_instance(inputs, Union[None, str, list[str]])
        assert is_instance(update, Optional[bool])

        targets = as_list(targets, 'targets')
        inputs = as_list(inputs, 'inputs')

        # ensure that inputs DOESN'T contain targets
        common = set(inputs).intersection(targets)
        if len(common) > 0:
            self.log.warning(f"'inputs' columns contain some 'target' columns: {common}.Removed from 'inputs'")
            inputs = list(set(inputs).difference(targets))

        now = datetime.now()
        ntargets = len(targets)

        data_model_id = self._convert_id(name, self.iDataModelMaster, ['description'], nullable=True)
        already_exists = data_model_id is not None
        if already_exists and not update:
            self.log.warning(f"Data Model {name} already existent")
            return IDataModel(self, data_model_id, self.iDataModelMaster)

        if already_exists:
            self.log.warning(f"Delete Data Model {name}")
            self.delete_data_model(name)

        with self.engine.connect() as conn:
            # 1) create data model master
            table = self.iDataModelMaster
            query = insert(table).values(
                description=name,
            ).returning(table.c.id)
            self.log.debug(query)
            data_model_id: int = conn.execute(query).scalar()

            # 2) create data model detail
            table = self.iDataModelDetail
            count = 0
            for measure in targets + inputs:
                query = insert(table).values(
                    measure_id=measure,
                    leaf_formula=None,
                    non_leaf_formula=None,
                    type='FEED' if count < ntargets else 'INPUT',
                    non_leaf_type='AGGREGATION',
                    created_date=now,
                    roll='N',
                    data_model_id_fk=data_model_id,
                    description=measure,
                    skills=None,
                    skill_enabled='Y',
                    popup_id=None,
                    default_value=0,
                    positive_only='N',
                    model_percision=None,   # wrong 'model_precision'
                    measure_mode='PLAN',
                    linked_measure=None,
                    period_agg_type=None
                ).returning(table.c.id)
                if count == 0: self.log.debug(query)
                measure_id: int = conn.execute(query).scalar()
                count += 1
            conn.commit()
        return IDataModel(self, data_model_id, self.iDataModelMaster)
    # end

    # -----------------------------------------------------------------------
    # Data Master

    def data_master(self, id: Union[int, str]) -> IDataMaster:
        data_master_id = self._convert_id(id, self.iDataMaster, ['description'])
        return IDataMaster(self, data_master_id, self.iDataMaster)

    def find_data_master(self, data_model: Union[int, str],
                         area_hierarchy: Union[int, str], skill_hierarchy: Union[int, str]) -> Optional[IDataMaster]:
        """
        Find a Data Master having the selected Data Model, Area Hierarchy, Skill Hierarchy (and Period Hierarchy)
        Note: if there are multiple Data Masters, it is selected the first one.

        :param data_model: Data Model
        :param area_hierarchy: Area Hierarchy
        :param skill_hierarchy: Skill Hierarchy
        :return: Data Model or None
        """

        data_model_id = self._convert_id(data_model, self.iDataModelMaster, ['description'])
        area_hierarchy_id = self._convert_id(area_hierarchy, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        skill_hierarchy_id = self._convert_id(skill_hierarchy, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])

        with self.engine.connect() as conn:
            table = self.iDataMaster
            query = select(table.c.id).distinct().where((table.c['area_id_fk'] == area_hierarchy_id) &
                                                        (table.c['skill_id_fk'] == skill_hierarchy_id) &
                                                        (table.c['idatamodel_id_fk'] == data_model_id))
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            if len(rlist) == 1:
                return IDataMaster(self.ipom, to_data(rlist[0]), table)
            elif len(rlist) == 0:
                self.log.error(f"No Data Master found with ({data_model_id},{area_hierarchy_id},{skill_hierarchy_id})")
                return None
            else:
                self.log.error(
                    f"Multiple Data Masters with found with (dara_model:{data_model_id},area_hierarchy:{area_hierarchy_id},skill_hierarchy:{skill_hierarchy_id})")
                return IDataMaster(self.ipom, to_data(rlist[-1]), table)

    def create_data_master(self, name: str, *,
                           data_model: Union[int, str],
                           area_hierarchy: Union[int, str],
                           skill_hierarchy: Union[int, str],
                           period_hierarchy: Literal['day', 'week', 'month'] = 'day',
                           periods: int = 90,
                           update: Optional[bool] = None) -> IDataMaster:
        """
        Create a Data Master

        :param name: name of the Data Master
        :param data_model: Data Model to use
        :param area_hierarchy: Area Hierarchy to use
        :param skill_hierarchy: Skill Hirerachy to use
        :param period_hierarchy: Period Hierarchy to use
        :param periods: period length to use
        :param update: how to update the data already present in the database
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        :return:
        """
        assert is_instance(period_hierarchy, Literal['day', 'week', 'month'])
        assert is_instance(periods, int) and periods > 0

        # data_model_id = self._convert_id(data_model, self.iDataModelMaster, ['description'])
        # area_hierarchy_id = self._convert_id(area_hierarchy, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        # skill_hierarchy_id = self._convert_id(skill_hierarchy, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        # assert self.hierarchy_type(area_hierarchy_id) == 'area'
        # assert self.hierarchy_type(skill_hierarchy_id) == 'skill'

        data_model_id = self.data_model(data_model).id
        area_hierarchy_id = self.area_hierarchy(area_hierarchy).id
        skill_hierarchy_id = self.skill_hierarchy(skill_hierarchy).id

        table = self.iDataMaster
        data_master_id = self._convert_id(name, table, ['description'], nullable=True)
        already_exists = data_master_id is not None
        if already_exists and not update:
            self.log.warning(f"Data Master {name} already existent")
            return IDataMaster(self, data_master_id, table)

        if already_exists:
            self.log.warning(f"Delete Data Master {name}")
            self.delete_data_master(data_master_id)

        with self.engine.connect() as conn:
            if already_exists:
                from sqlalchemy import update
                query = update(table) \
                    .where(table.c['description'] == name) \
                    .values(
                        description=name,
                        area_id_fk=area_hierarchy_id,
                        skill_id_fk=skill_hierarchy_id,
                        idatamodel_id_fk=data_model_id,
                        period_hierarchy=period_hierarchy,
                        period=periods,
                        rule_enabled=True,
                        baseline_enabled=False,
                        opti_enabled=False
                    ) \
                    .returning(table.c.id)
            else:
                query = insert(table) \
                    .values(
                        description=name,
                        area_id_fk=area_hierarchy_id,
                        skill_id_fk=skill_hierarchy_id,
                        idatamodel_id_fk=data_model_id,
                        period_hierarchy=period_hierarchy,
                        period=periods,
                        rule_enabled='Y',
                        baseline_enabled='N',
                        opti_enabled='N') \
                    .returning(table.c.id)
            self.log.debug(query)
            data_master_id = conn.execute(query).scalar()
            conn.commit()
        return IDataMaster(self, data_master_id, table)
    # end

    def delete_data_master(self, id: Union[int, str]):
        data_master_id = self._convert_id(id, self.iDataMaster, ['description'], nullable=True)
        if data_master_id is None:
            return

        with self.engine.connect() as conn:
            # 1) delete dependencies
            table = self.iDataValuesMaster
            query = delete(table).where(table.c['idata_master_fk'] == data_master_id)
            self.log.debug(query)
            conn.execute(query)

            # 2) delete data master
            table = self.iDataMaster
            query = delete(table).where(table.c.id == data_master_id)
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
    # end

    # def select_data_master_ids(self, data_model_id: int, area_hierarchy_id: int, skill_hierarchy_id: int) \
    #         -> list[int]:
    #     with self.engine.connect() as conn:
    #         table = self.ipom.iDataMaster
    #         query = select(table.c.id).where((table.c['area_id_fk'] == area_hierarchy_id) &
    #                                          (table.c['skill_id_fk'] == skill_hierarchy_id) &
    #                                          (table.c['idatamodel_id_fk'] == data_model_id))
    #         self.log.debug(f"{query}")
    #         rlist = conn.execute(query).fetchall()
    #         return [result[0] for result in rlist]

    # -----------------------------------------------------------------------
    # Prediction Plan

    @deprecated(reason="It is better to use 'prediction_plan(name, data_master)'")
    def prediction_plans(self) -> IPredictionPlans:
        return IPredictionPlans(self)

    def prediction_plan(self, name: str, data_master: Union[int, str]) -> IPredictionPlan:
        return IPredictionPlan(self, name, data_master)

    def delete_prediction_plan(self, id: Union[int, str], data_master: Union[int, str]):
        self.prediction_plan(id, data_master).delete()

    def create_prediction_plan(
            self, name: str, data_master: Union[int, str], *,
            start_date: datetime, end_date: Optional[datetime] = None,
            periods: Optional[int] = None, note: Optional[str]=None,
            update: Optional[bool] = None) -> IPredictionPlan:

        assert is_instance(name, str)
        assert is_instance(data_master, Union[int, str])
        assert is_instance(start_date, datetime)
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])
        assert is_instance(note, Optional[str])

        pplan = self.prediction_plan(name, data_master)
        return pplan.create(
            start_date, end_date, periods, note, update
        )
    # end

    # -----------------------------------------------------------------------

    def data_values_master(self, id: Union[int, str]) -> IDataValuesMaster:
        id = self._convert_id(id, self.iDataValuesMaster, ['name'])
        return IDataValuesMaster(self, id, self.iDataValuesMaster)

    def select_data_values(self, data_values_master_id) -> DataFrame:
        with self.engine.connect() as conn:
            table = self.iDataValuesDetail
            query = select(table.c['state_date', 'model_detail_id_fk', 'skill_id_fk', 'value']) \
                .where(table.c['value_master_fk'] == data_values_master_id)
            self.log.debug(query)
            df = pd.read_sql_query(query, self.engine)
        return df

    def select_plan_ids(
        self,
        name: Optional[str],
        data_master_ids: list[int],
        area_feature_ids: list[int],
    ) -> list[int]:
        return self.select_data_values_master_ids(name, data_master_ids, area_feature_ids)

    def select_data_values_master_ids(
        self,
        name: Optional[str],
        data_master_ids: list[int],
        area_feature_ids: list[int],
    ) -> list[int]:
        # alias: select_plan_ids(...)
        table = self.iDataValuesMaster
        if name is None:
            query = select(table.c.id).where(
                table.c['idata_master_fk'].in_(data_master_ids) &
                table.c['area_id'].in_(area_feature_ids)
            )
            self.log.debug(query)
        else:
            query = select(table.c.id).where(
                (table.c['name'] == name) &
                table.c['idata_master_fk'].in_(data_master_ids) &
                table.c['area_id'].in_(area_feature_ids)
            )
            self.log.debug(query)
        with self.engine.connect() as conn:
            rlist = conn.execute(query)
            return [result[0] for result in rlist]

    def select_data_values_master_date_interval(self, plan_ids: list[int]) \
            -> Optional[tuple[datetime, datetime]]:
        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                table.c.id.in_(plan_ids)
            )
            self.log.debug(query)
            result = conn.execute(query).fetchone()
            if result[0] is None or result[1] is None:
                return None
            else:
                return result[0], result[1]

    def select_measure_names(self, measure_ids: list[int]) -> dict[int, str]:
        mdict = {}
        with self.engine.connect() as conn:
            tmeasure = self.iDataModelDetail
            query = select(tmeasure.c['id', 'measure_id']).where(tmeasure.c['id'].in_(measure_ids))
            self.log.debug(query)
            rlist = conn.execute(query)
            for id, name in rlist:
                mdict[id] = name
        return mdict

    # -----------------------------------------------------------------------
    # Time Series Focussed

    def delete_time_series_focussed(self, id: Union[int, str]):
        tsf_id = self._convert_id(id, self.iPredictMasterFocussed, ['ipr_conf_master_name', 'ipr_conf_master_desc'],
                                  nullable=True)
        if tsf_id is None:
            return

        with self.engine.connect() as conn:
            # 1) delete tb_ipr_conf_detail_focussed
            table = self.iPredictDetailFocussed
            query = delete(table).where(table.c['ipr_conf_master_id'] == tsf_id)
            self.log.debug(query)
            conn.execute(query)

            # 2) delete tb_ipr_conf_master_focussed
            table = self.iPredictMasterFocussed
            query = delete(table).where(table.c['id'] == tsf_id)
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return
    # end

    def create_time_series_focussed(self, name: str, *,
                                    targets: Union[str, list[str]],
                                    inputs: Union[None, str, list[str]] = None,
                                    data_master: Union[None, int, str] = None,
                                    # data_model: Union[None, int, str] = None,
                                    # area_hierarchy: Union[None, int, str] = None,
                                    # skill_hierarchy: Union[None, int, str] = None,
                                    description: Optional[str] = None,
                                    update: Optional[bool] = None) -> IPredictTimeSeries:
        """
        Create a time series

        :param name: Time Series name
        :param targets: list of target measures
        :param inputs: list of input measures
        :param data_master: Data Master, alternative to (data_model, area_hierarchy, skill_hierarchy)
        :param description:  Time Series description
        :param update:
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        """
        # :param data_model: Data Model (alternative to data_master)
        # :param area_hierarchy: Area Hierarchy (alternative to data_master)
        # :param skill_hierarchy: Skill Hierarchy (alternative to data_master)

        assert is_instance(name, str)
        assert is_instance(targets, Union[str, list[str]])
        assert is_instance(inputs, Union[None, str, list[str]])
        assert is_instance(data_master, Union[None, int, str])
        assert is_instance(description, Optional[str])
        assert is_instance(update, Optional[bool])
        # assert is_instance(data_model, Union[None, int, str])
        # assert is_instance(area_hierarchy, Union[None, int, str])
        # assert is_instance(skill_hierarchy, Union[None, int, str])

        data_master_id = data_master
        data_master = self.data_master(data_master_id)
        data_master_id = data_master.id
        data_model: IDataModel = data_master.data_model
        data_model_id = data_model.id
        area_hierarchy_id = data_master.area_hierarchy.id
        skill_hierarchy_id = data_master.skill_hierarchy.id

        targets = as_list(targets, 'targets')
        inputs = as_list(inputs, 'inputs')
        description = name if description is None else description

        # if data_master is not None:
        #     data_master_id = data_master
        #     data_master = self.data_master(data_master_id)
        #     data_model_id = data_master.data_model.id
        #     area_hierarchy_id = data_master.area_hierarchy.id
        #     skill_hierarchy_id = data_master.skill_hierarchy.id
        # else:
        #     data_model_id = self.data_model(data_model).id
        #     area_hierarchy_id = self.area_hierarchy(area_hierarchy).id
        #     skill_hierarchy_id = self.skill_hierarchy(skill_hierarchy).id
        # end

        table = self.iPredictMasterFocussed
        tsf_id = self._convert_id(name, table, ['ipr_conf_master_name', 'ipr_conf_master_desc'], nullable=True)
        already_exists = tsf_id is not None
        if already_exists and not update:
            self.log.warning(f"Time Series {name} already existent")
            return IPredictTimeSeries(self, tsf_id, data_master_id,)

        if already_exists:
            self.log.warning(f"Delete Time Series {name}")
            self.delete_time_series_focussed(name)

        # create the tb_ipr_conf_master_focussed
        with self.engine.connect() as conn:
            # 1) fill tb_ipr_conf_master_focussed
            table = self.iPredictMasterFocussed
            query = insert(table).values(
                ipr_conf_master_name=name,
                ipr_conf_master_desc=description,
                idata_model_details_id_fk=data_model_id,
                area_id_fk=area_hierarchy_id,
                skill_id_fk=skill_hierarchy_id,
                idata_id_fk=None
            ).returning(table.c.id)
            self.log.debug(query)
            tsf_id = conn.execute(query).scalar()
            # 2) fill tb_ipr_conf_detail_focussed
            table = self.iPredictDetailFocussed
            for measure in targets:
                measure_id = data_model.measure(measure).id
                query = insert(table).values(
                    parameter_desc=measure,
                    parameter_value='output',
                    ipr_conf_master_id=tsf_id,
                    parameter_id=measure_id,
                    skill_id_fk=skill_hierarchy_id,
                    to_populate=None,
                    period=None
                ).returning(table.c.id)
                target_id = conn.execute(query).scalar()
            # end
            for measure in inputs:
                measure_id = data_model.measure(measure).id
                query = insert(table).values(
                    parameter_desc=measure,
                    parameter_value='input',
                    ipr_conf_master_id=tsf_id,
                    parameter_id=measure_id,
                    skill_id_fk=skill_hierarchy_id,
                    to_populate=None,
                    period=None
                ).returning(table.c.id)
                input_id = conn.execute(query).scalar()
            # end
            conn.commit()
        return IPredictTimeSeries(self, tsf_id, data_master_id,)

    # alias
    def time_series_focussed(self,
                             id: Union[int, str],
                             data_master: Union[None, int, str] = None,
                             plan: Union[None, int, str] = None) -> IPredictTimeSeries:
        """
        Time series focussed (tables 'tb_ipr_conf_master_focussed', tb_ipr_conf_detail_focussed')

        Note: the preferred way to pass a plan is using the name

        :param id: time series id or name
        :param data_master: data master id or name
        :param plan: plan id or name
        :return: an IPredictTimeSeries object
        """
        assert is_instance(id, Union[int, str])
        assert is_instance(data_master, Union[None, int, str])
        assert is_instance(plan, Union[None, int, str])
        assert data_master is not None or plan is not None, f"Invalid time series: missing 'data_master' or 'plan' parameter"

        # convert plan into an integer, if it is passed an int value as string
        plan = safe_int(plan)

        pmf: IPredictMasterFocussed = self.predict_master_focussed(id)

        area_feature_dict = pmf.area_hierarchy.feature_ids(with_name=True)
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_dict = pmf.skill_hierarchy.feature_ids(with_name=True)
        skill_feature_ids = list(skill_feature_dict.keys())

        # retrieve the plan name
        if isinstance(plan, int):
            plan = self.data_values_master(plan).name

        if data_master is not None:
            data_master_id: int = self.data_master(data_master).id
        else:
            _, data_master_id = self._select_data_values_master_ids_by_plan(plan, area_feature_ids)

        return IPredictTimeSeries(self, pmf.id, data_master_id)

    def predict_master_focussed(self, id: Union[int, str]) -> IPredictMasterFocussed:
        id = self._convert_id(id, self.iPredictMasterFocussed, ['ipr_conf_master_name', 'ipr_conf_master_desc'])
        return IPredictMasterFocussed(self, id, self.iPredictMasterFocussed)

    # -----------------------------------------------------------------------
    # data_values_detail_hist
    # data_values_detail

    def save_area_skill_train_data(
        self,
        area_feature_id: int, skill_feature_id: int, measure_id: int, plan_id: int,
        df: DataFrame, update: Optional[bool] = None):
        """

        :param area_feature_id:
        :param skill_feature_id:
        :param measure_id:
        :param plan_id:
        :param df:
        :param update: how to update the data already present in the database
            - None: all data is deleted and replaced
                    (delete and insert)
            - True: the data in the dataset replaces the same data in the database
                    (update or insert)
            - False:  all data in the database is not deleted or updated
                    (insert only):return:
        """

        assert is_instance(area_feature_id, int)
        assert is_instance(skill_feature_id, int)
        assert is_instance(measure_id, int)
        assert is_instance(df, DataFrame)
        assert is_instance(plan_id, int)

        start_date = pdx.to_datetime(df['state_date'].min())
        end_date = pdx.to_datetime(df['state_date'].max())
        n = len(df)

        now = datetime.now()

        table = self.iDataValuesDetailHist
        with self.engine.connect() as conn:
            if update is None:
                # ERROR: it seems to not work!
                # query = delete(table).where(
                #     (table.c['state_date'] >= start_date) &
                #     (table.c['state_date'] <= end_date) &
                #     (table.c['model_detail_id_fk'] == measure_id) &
                #     (table.c['area_id_fk'] == area_feature_id) &
                #     (table.c['skill_id_fk'] == skill_feature_id)
                # )
                # self.log.debug(query)
                # conn.execute(query)

                query = text("""
                DELETE FROM tb_idata_values_detail_hist
                 WHERE state_date >= :start_date
                   AND state_date <= :end_date
                   AND model_detail_id_fk = :measure_id
                   AND area_id_fk = :area_feature_id
                   AND skill_id_fk = :skill_feature_id
                """)
                try:
                    conn.execute(query, parameters=dict(
                        start_date=start_date,
                        end_date=end_date,
                        measure_id=measure_id,
                        area_feature_id=area_feature_id,
                        skill_feature_id=skill_feature_id
                    ))
                except Exception as e:
                    pass

            # TODO: BETTER IMPLEMENTATION with 'prepared_statement'
            # for i in range(n):
            #     state_date = pdx.to_datetime(df['state_date'].iloc[i])
            #     value = float(df[measure_id].iloc[i])
            #     query = insert(table).values(
            #         area_id_fk=area_feature_id,
            #         skill_id_fk=skill_feature_id,
            #         model_detail_id_fk=measure_id,
            #         state_date=state_date,
            #         value=value,
            #
            #         value_master_fk=plan_id,
            #         updated_date=now,
            #         value_type=None,
            #         value_insert_time=None,
            #     )
            #     # self.log.debug(query)
            #     conn.execute(query)

            bulk_data = [
                dict(area_id_fk=area_feature_id,
                     skill_id_fk=skill_feature_id,
                     model_detail_id_fk=measure_id,
                     state_date=pdx.to_datetime(df['state_date'].iloc[i]),
                     value=float(df[measure_id].iloc[i]),

                     value_master_fk=plan_id,
                     updated_date=now,
                     value_type=None,
                     value_insert_time=None,
                )
                for i in range(n)
            ]
            conn.execute(table.insert(), bulk_data)
            conn.commit()
        return

    def save_area_skill_predict_data(
        self,
        area_feature_id: int, skill_feature_id: int, measure_id: int, plan_id: int,
        df: DataFrame, update: Optional[bool] = None):
        """

        :param area_feature_id:
        :param skill_feature_id:
        :param measure_id:
        :param plan_id:
        :param df:
        :param update: how to update the data already present in the database
            - None: all data is deleted and replaced
                    (delete and insert)
            - True: the data in the dataset replaces the same data in the database
                    (update or insert)
            - False:  all data in the database is not deleted or updated
                    (insert only)
        """
        assert is_instance(area_feature_id, int)
        assert is_instance(skill_feature_id, int)
        assert is_instance(measure_id, int)
        assert is_instance(plan_id, int)
        assert is_instance(df, DataFrame)

        start_date = pdx.to_datetime(df['state_date'].min())
        end_date = pdx.to_datetime(df['state_date'].max())
        n = len(df)

        now = datetime.now()

        table = self.iDataValuesDetail
        with self.engine.connect() as conn:
            if update is None:
                # ERROR: it seems to not work!
                # query = delete(table).where(
                #     (table.c['state_date'] >= start_date) &
                #     (table.c['state_date'] <= end_date) &
                #     (table.c['model_detail_id_fk'] == measure_id) &
                #     (table.c['area_id_fk'] == area_feature_id) &
                #     (table.c['skill_id_fk'] == skill_feature_id)
                # )
                # self.log.debug(query)
                # conn.execute(query)

                query = text("""
                DELETE FROM tb_idata_values_detail
                 WHERE state_date >= :start_date
                   AND state_date <= :end_date
                   AND value_master_fk = :plan_id
                   AND model_detail_id_fk = :measure_id
                   AND skill_id_fk = :skill_feature_id
                """)
                try:
                    conn.execute(query, parameters=dict(
                        start_date=start_date,
                        end_date=end_date,
                        plan_id=plan_id,
                        area_feature_id=area_feature_id,
                        skill_feature_id=skill_feature_id,
                        measure_id=measure_id,
                    ))
                except Exception as e:
                    pass

            # TODO: BETTER IMPLEMENTATION with 'prepared_statement'
            # for i in range(n):
            #     state_date = pdx.to_datetime(df['state_date'].iloc[i])
            #     value = float(df[measure_id].iloc[i])
            #     query = insert(table).values(
            #         value_master_fk=plan_id,
            #         state_date=state_date,
            #         skill_id_fk=skill_feature_id,
            #         model_detail_id_fk=measure_id,
            #         value=value,
            #         updated_date=now,
            #     )
            #     # self.log.debug(query)
            #     conn.execute(query)

            bulk_data = [
                dict(value_master_fk=plan_id,
                     state_date=pdx.to_datetime(df['state_date'].iloc[i]),
                     skill_id_fk=skill_feature_id,
                     model_detail_id_fk=measure_id,
                     value=float(df[measure_id].iloc[i]),
                     updated_date=now,)
                for i in range(n)
            ]
            conn.execute(table.insert(), bulk_data)
            conn.commit()
        return

    # -----------------------------------------------------------------------
    # [tb_idata_values_master]
    # id
    # start_date
    # end_date
    # name
    # created_date
    # idata_master_fk       -> [td_idata_master]
    # area_id               -> [tb_attribute_detail]
    # loan_updated_time
    # published
    # isscenario
    # temp_ind
    # last_updated_date
    # published_id
    # note
    #
    # [tb_idata_values_detail_hist]
    # id
    # value_master_fk       -> [tb_idata_value_master]
    # state_date            // value timestamp
    # updated_date          // when the data was inserted in the database
    # model_detail_id_fk    -> [tb_idata_model_detail] (measure)
    # area_id_fk            -> [tb_attribute_detail]   (area feature)
    # skill_id_fk           -> [tb_attribute_detail]   (skill feature)
    # value                 // measure value
    # value_type            // [NULL]
    # value_insert_time     // [NULL]
    #
    # [tb_idata_values_detail]
    # id
    # value_master_fk       -> [tb_idata_value_master]
    # state_date            // value timestamp
    # updated_date          // when the data was inserted in the database
    # model_detail_id_fk    -> [tb_idata_model_detail] (measure)
    # skill_id_fk           -> [tb_attribute_detail]   (skill feature)
    # value                 // measure value
    #

    def delete_train_data(self,
                          data_master_id: int,
                          plan_ids: list[int],
                          area_feature_dict: dict[int, str],
                          skill_feature_dict: dict[int, str],
                          measure_dict: dict[int, str]
                          ):

        assert is_instance(data_master_id, int)
        assert is_instance(plan_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(measure_dict, dict[int, str])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        with self.engine.connect() as conn:
            table = self.iDataValuesDetailHist

            # 2) retrieve the data with 'skill NOT NULL'
            query = delete(table) \
                .where(table.c['value_master_fk'].in_(plan_ids) &
                       table.c['model_detail_id_fk'].in_(measure_ids) &
                       table.c['area_id_fk'].in_(area_feature_ids) &
                       table.c['skill_id_fk'].in_(skill_feature_ids))
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return
    # end

    def delete_predict_data(self,
                            data_master_id: int,
                            plan_ids: list[int],
                            area_feature_dict: dict[int, str],
                            skill_feature_dict: dict[int, str],
                            measure_dict: dict[int, str],
                            ):

        assert is_instance(data_master_id, int)
        assert is_instance(plan_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(measure_dict, dict[int, str])
        # assert is_instance(start_date, Optional[datetime])
        # assert is_instance(end_date, Optional[datetime])
        # assert is_instance(freq, Literal['D', 'W', 'M'])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        with self.engine.connect() as conn:
            table = self.iDataValuesDetail
            query = delete(table) \
                .where(table.c['value_master_fk'].in_(plan_ids) &
                       table.c['model_detail_id_fk'].in_(measure_ids) &
                       table.c['skill_id_fk'].in_(skill_feature_ids))
            self.log.debug(query)
            conn.execute(query, parameters=dict(
                plan_ids=tuple(plan_ids),
                measure_ids=tuple(measure_ids),
                skill_feature_ids=tuple(skill_feature_ids),
                area_feature_ids=tuple(area_feature_ids)
            ))
            conn.commit()
        return
    # end

    def select_train_data(
        self,
        data_master_id: int,
        plan_ids: list[int],  # data_values_master_ids
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        measure_dict: dict[int, str],
        new_format=False) -> DataFrame:
        """
        Retrieve the historical data from 'tb_idata_values_detail_hist' based on

            - data_master_id
            - plan_ids
            - area_feature_ids
            - skill_feature_ids
            - measure_ids

        It is possible to replace the area/skill/measure ids with the correspondent names

        :param data_master_id:
        :param plan_ids:
        :param area_feature_dict:
        :param skill_feature_dict:
        :param measure_dict:
        :param new_format: if to create a dataframe compatible with
            the current implementation of the new format
        :return: a dataframe with the following columns
                if 'new_format == True':
                    columns: ['area:str', 'skill:str', 'date:datetime', <measure_1:float>, ...]
                else
                    columns: ['skill_id_fk:int', 'area_id_fk:int', 'time:datetime', 'day:str', <measure_1: float>, ...]
        """

        assert is_instance(data_master_id, int)
        assert is_instance(plan_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(measure_dict, dict[int, str])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        table = self.iDataValuesDetailHist

        # 2) retrieve the data with 'skill NOT NULL'
        query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
            .where(table.c['model_detail_id_fk'].in_(measure_ids) &
                   table.c['area_id_fk'].in_(area_feature_ids) &
                   table.c['skill_id_fk'].in_(skill_feature_ids))
        self.log.debug(query)
        df_with_skill = pd.read_sql_query(query, self.engine)

        # 3) retrieve the data with 'skill IS NULL'
        query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
            .where(table.c['model_detail_id_fk'].in_(measure_ids) &
                   table.c['area_id_fk'].in_(area_feature_ids) &
                   (table.c['skill_id_fk'] == None)
                   )
        self.log.debug(query)
        df_no_skill = pd.read_sql_query(query, self.engine)

        # 4) concatenate df_with_skill WITH df_no_skill
        df = concatenate_no_skill_df(df_with_skill, df_no_skill, skill_feature_dict)

        return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)
    # end

    def select_predict_data(
        self,
        data_master_id: int,
        plan_ids: list[int],  # data_values_master_ids
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        measure_dict: dict[int, str],
        new_format=False) -> DataFrame:
        """

        :param data_master_id:
        :param plan_ids:
        :param area_feature_dict:
        :param skill_feature_dict:
        :param measure_dict:
        :param new_format:
        :return:
        """

        assert is_instance(data_master_id, int)
        assert is_instance(plan_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(measure_dict, dict[int, str])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        # 2) retrieve the data with 'skill NOT NULL'
        qtext = """
                select tivm.area_id as area_id_fk,
                       tivd.skill_id_fk as skill_id_fk,
                       tivd.model_detail_id_fk as model_detail_id_fk,
                       tivd.state_date as state_date,
                       tivd.value as value
                 from tb_idata_values_detail as tivd
                 join tb_idata_values_master as tivm on tivm.id = tivd.value_master_fk
                where tivd.value_master_fk in :plan_ids
                  and tivd.model_detail_id_fk in :measure_ids
                  and tivd.skill_id_fk in :skill_feature_ids
                  and tivm.area_id in :area_feature_ids
                """
        query = text(qtext)
        self.log.debug(query)
        df = pd.read_sql_query(query, self.engine, params=dict(
            plan_ids=tuple(plan_ids),
            measure_ids=tuple(measure_ids),
            skill_feature_ids=tuple(skill_feature_ids),
            area_feature_ids=tuple(area_feature_ids)
        ))

        return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)

    def select_predict_data_ext(
        self,
        data_master_id: int,
        plan_ids: list[int],  # data_values_master_ids
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        input_measure_ids: list[int],
        measure_dict: dict[int, str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        freq: Literal['D', 'W', 'M'] = 'D',
        new_format=False) -> DataFrame:
        """

        :param data_master_id:
        :param plan_ids:
        :param area_feature_dict:
        :param skill_feature_dict:
        :param input_measure_ids:
        :param measure_dict:
        :param start_date:
        :param end_date:
        :param freq:
        :param new_format:
        :return:
        """

        assert is_instance(data_master_id, int)
        assert is_instance(plan_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(input_measure_ids, list[int])
        assert is_instance(measure_dict, dict[int, str])
        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(freq, Literal['D', 'W', 'M'])

        # Note: the dataset contains all measures in 'input_measure_ids' or 'measure_dict'
        #   plus: 'area', 'skill', 'date'

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        # 2) retrieve start/end dates for each area
        start_end_date_dict = self._select_start_end_date_dict(plan_ids, area_feature_ids)

        # add the DEFAULT start/end date for the areas without a date range
        # Note: it is used 0 (ZERO) as key
        if start_date is not None and end_date is not None:
            start_end_date_dict[0] = (start_date, end_date)

        # Note: [tb_idata_values_detail]
        #   DOESNT' CONTAIN 'area_id_fk'
        #   BUT it has a reference with [tb_idata_values_master] ('value_master_fk)
        #   AND 'tb_idata_values_master' contains 'area_id', that is, the required 'area_id_fk'
        df = self._select_predict_data(plan_ids, area_feature_ids, skill_feature_ids, measure_ids)

        df_pivoted = pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)

        df_pivoted = compose_predict_df(
            df_pivoted,
            area_feature_dict,
            skill_feature_dict,
            input_measure_ids,
            measure_dict,
            start_end_date_dict,
            freq,
            new_format=new_format
        )

        return df_pivoted
    # end

    # -----------------------------------------------------------------------

    def _select_start_end_date_dict(self, plan_ids, area_feature_ids) \
            -> dict[int, tuple[datetime, datetime]]:
        # data_values_master_ids -> plan_ids
        # 2) retrieve start/end dates for each area

        # qtext = """
        #         select tivm.area_id as area_id_fk, tivm.start_date, tivm.end_date
        #           from tb_idata_values_master as tivm
        #          where tivm.id in :plan_ids
        #            and tivm.area_id in :area_feature_ids
        #         """
        # query = text(qtext)

        table = self.iDataValuesMaster
        query = select(table.c['area_id', 'start_date', 'end_date']).where(
            table.c.id.in_(plan_ids) &
            table.c['area_id'].in_(area_feature_ids)
        )
        self.log.debug(query)
        midnight = time(0, 0, 0)

        with self.engine.connect() as conn:
            rlist = conn.execute(query, parameters=dict(
                plan_ids=tuple(plan_ids),
                area_feature_ids=tuple(area_feature_ids)
            )).fetchall()
            start_end_date_dict = {
                r[0]: (
                    datetime.combine(r[1], midnight),
                    datetime.combine(r[2], midnight)
                )
                for r in rlist
            }
        return start_end_date_dict
    # end

    def _select_predict_data(self,
                             plan_ids: list[int],  # data_values_master_ids
                             area_feature_ids: list[int],
                             skill_feature_ids: list[int],
                             measure_ids: list[int]) \
            -> DataFrame:
        # qtext = """
        #         select tivm.area_id as area_id_fk,
        #                tivd.skill_id_fk as skill_id_fk,
        #                tivd.model_detail_id_fk as model_detail_id_fk,
        #                tivd.state_date as state_date,
        #                tivd.value as value
        #          from tb_idata_values_detail as tivd,
        #               tb_idata_values_master as tivm
        #         where tivd.value_master_fk in :plan_ids
        #           and tivd.model_detail_id_fk in :measure_ids
        #           and tivd.skill_id_fk in :skill_feature_ids
        #           and tivm.area_id in :area_feature_ids
        #           and tivm.id in :plan_ids
        #           and tivm.id = tivd.value_master_fk
        #         """
        qtext = """
                select tivm.area_id as area_id_fk,
                       tivd.skill_id_fk as skill_id_fk,
                       tivd.model_detail_id_fk as model_detail_id_fk,
                       tivd.state_date as state_date,
                       tivd.value as value
                 from tb_idata_values_detail as tivd
                 join tb_idata_values_master as tivm on tivm.id = tivd.value_master_fk
                where tivd.value_master_fk in :plan_ids
                  and tivd.model_detail_id_fk in :measure_ids
                  and tivd.skill_id_fk in :skill_feature_ids
                  and tivm.area_id in :area_feature_ids
        """
        query = text(qtext)
        self.log.debug(query)
        df = pd.read_sql_query(query, self.engine, params=dict(
            plan_ids=tuple(plan_ids),
            measure_ids=tuple(measure_ids),
            skill_feature_ids=tuple(skill_feature_ids),
            area_feature_ids=tuple(area_feature_ids)
        ))
        return df
    # end

    def _select_data_values_master_ids(self, data_master_ids: list[int], area_feature_ids: list[int]) \
            -> list[int]:
        assert is_instance(data_master_ids, list[int])
        assert is_instance(area_feature_ids, list[int])

        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(table.c.id).distinct().where(
                table.c['idata_master_fk'].in_(data_master_ids) &
                table.c['area_id'].in_(area_feature_ids)
            )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            return [result[0] for result in rlist]
    # end

    def _select_data_values_master_ids_by_plan(self, plan_name: str, area_feature_ids: list[int]) \
            -> tuple[list[int], list[int]]:
        assert is_instance(plan_name, str)
        assert is_instance(area_feature_ids, list[int])

        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(table.c.id, table.c['idata_master_fk']).distinct().where(
                (table.c['name'] == plan_name) &
                table.c['area_id'].in_(area_feature_ids)
            )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            data_master_ids = list({result[1] for result in rlist})
            plan_ids = list({result[0] for result in rlist})
            # data_values_master_ids -> plan_ids

        if len(data_master_ids) > 1:
            self.log.warning(f"Multiple Data Masters for plan {plan_name}")

        return plan_ids, data_master_ids[-1]
    # end

    # -----------------------------------------------------------------------

    def clear_predict_focussed_data(self, predict_master_id: int):
        # clear the content of 'tb_ipr_model_detail_focussed' and 'tb_ipr_train_data_focussed'
        with self.engine.connect() as conn:
            table = self.iPredictModelDetailFocussed
            query = delete(table).where(table.c['ipr_conf_master_id_fk'] == predict_master_id)
            self.log.debug(query)
            conn.execute(query)

            table = self.iPredictTrainDataFocussed
            query = delete(table).where(table.c['ipr_conf_master_id_fk'] == predict_master_id)
            self.log.debug(query)
            conn.execute(query)
            conn.commit()

    # end

    # -----------------------------------------------------------------------

    def _load_metadata(self):
        self.metadata = MetaData()

        # -------------------------------------------------------------------
        # Main data
        # -------------------------------------------------------------------

        # iData Model
        self.iDataModelMaster \
            = Table('tb_idata_model_master', self.metadata, autoload_with=self.engine)
        # [tb_idata_model_master]
        # id (bigint)
        # description (varchar(256))

        self.iDataModelDetail \
            = Table('tb_idata_model_detail', self.metadata, autoload_with=self.engine)
        # [tb_idata_model_detail]
        # id (bigint)
        # measure_id (varchar(256))
        # leaf_formula (text)
        # non_leaf_formula (text)
        # type (varchar(256))
        # non_leaf_type (varchar(256))
        # created_date (date)
        # roll (char(1))
        # data_model_id_fk (bigint)
        # description (varchar(256))
        # skills (varchar(256))
        # skill_enabled (varchar(256))
        # popup_id (bigint)
        # default_value (double precision)
        # positive_only (char(1))
        # model_percision (integer)
        # measure_mode (varchar(256))
        # linked_measure (varchar(256))
        # period_agg_type (varchar(256))

        # iData Module
        self.iDataModuleMaster \
            = Table('tb_idata_module_master', self.metadata, autoload_with=self.engine)
        # [tb_idata_module_master]
        # id (bigint)
        # module_id (varchar(20))
        # module_description (varchar(200))

        self.iDataModuleDetail \
            = Table('tb_idata_module_details', self.metadata, autoload_with=self.engine)
        # [tb_idata_module_details]
        # id (bigint)
        # master_id (bigint)
        # idata_id (bigint)
        # display_name (varchar(100))
        # is_enabled (varchar(1))

        # iData Master
        self.iDataMaster \
            = Table('tb_idata_master', self.metadata, autoload_with=self.engine)
        # [tb_idata_master]
        # id (bigint)
        # area_id_fk (bigint)
        # skill_id_fk (bigint)
        # idatamodel_id_fk (bigint)
        # period (bigint)
        # period_hierarchy (varchar(256))
        # description (varchar(256))
        # rule_enabled (char(1))
        # baseline_enabled (char(1))
        # opti_enabled (char(1))

        self.iDataDetail = None

        self.iDataValuesMaster \
            = Table('tb_idata_values_master', self.metadata, autoload_with=self.engine)
        # [tb_idata_values_master]
        # id (bigint)
        # start_date (date)
        # end_date (date)
        # name (varchar(256))
        # created_date (timestamp(6))
        # idata_master_fk (bigint)
        # loan_updated_time (timestamp)
        # published (char(1))
        # isscenario (char(1))
        # temp_ind (char(1))
        # area_id (bigint)
        # last_updated_date (timestamp)
        # published_id (bigint)
        # note (text)

        self.iDataValuesDetail \
            = Table('tb_idata_values_detail', self.metadata, autoload_with=self.engine)
        # [tb_idata_values_detail]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)

        self.iDataValuesDetailHistory \
            = Table('tb_idata_values_detail_hist', self.metadata, autoload_with=self.engine)
        # [tb_idata_values_detail_hist]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)
        # value_type (varchar(256))
        # value_insert_time (varchar(256))
        # area_id_fk (bigint)

        # Area/Skill Hierarchies
        self.AttributeMaster \
            = Table('tb_attribute_master', self.metadata, autoload_with=self.engine)
        # [tb_attribute_master]
        # id (bigint)
        # attribute_master_name (varchar(256))
        # attribute_desc (varchar(256))
        # createdby (varchar(256))
        # createddate (date)
        # hierarchy_type (bigint)

        self.AttributeDetail \
            = Table('tb_attribute_detail', self.metadata, autoload_with=self.engine)
        # [tb_attribute_detail
        # id (bigint)
        # attribute_master_id (bigint)
        # attribute (varchar(256))
        # description (varchar(256))
        # attribute_level (bigint)
        # parent_id (bigint)
        # createdby (varchar(256))
        # createddate (date)
        # is_leafattribute (boolean)

        # IPredict Focussed
        self.iPredictMasterFocussed \
            = Table('tb_ipr_conf_master_focussed', self.metadata, autoload_with=self.engine)
        # [tb_ipr_conf_master_focussed]
        # id (bigint)
        # ipr_conf_master_name (varchar(256))
        # ipr_conf_master_desc (varchar(256))
        # idata_model_details_id_fk (bigint)
        # area_id_fk (bigint)
        # skill_id_fk (bigint)
        # idata_id_fk (bigint)

        self.iPredictDetailFocussed \
            = Table('tb_ipr_conf_detail_focussed', self.metadata, autoload_with=self.engine)
        # [tb_ipr_conf_detail_focussed]
        # id (bigint)
        # parameter_desc (varchar(256))
        # parameter_value (varchar(256))
        # ipr_conf_master_id (bigint)
        # parameter_id (bigint)
        # to_populate (bigint)
        # skill_id_fk (bigint)
        # period (varchar(256))

        self.iPredictModelDetailFocussed \
            = Table('tb_ipr_model_detail_focussed', self.metadata, autoload_with=self.engine)
        # [tb_ipr_model_detail_focussed]
        # id(bigint)
        # best_model(text)
        # best_model_name(text)
        # best_r_2(double
        # precision)
        # best_wape(double
        # precision)
        # ohmodels_catftr(text)
        # area_id_fk(bigint)
        # ipr_conf_master_id_fk(bigint)
        # skill_id_fk(bigint)

        self.iPredictTrainDataFocussed \
            = Table('tb_ipr_train_data_focussed', self.metadata, autoload_with=self.engine)
        # [tb_ipr_train_data_focussed]
        # ipr_conf_master_id_fk (bigint)
        # area_id_fk (bigint)
        # skill_id_fk (bigint)
        # target (bigint)
        # actual (numeric)
        # predicted (numeric)
        # state_date (date)

        # -------------------------------------------------------------------
        # Values
        # -------------------------------------------------------------------
        # [tb_idata_values_detail]
        #   id (bigint)
        #   value_master_fk (bigint)
        #   state_date (date)
        #   updated_date (date)
        #   model_detail_id_fk (bigint)
        #   skill_id_fk (bigint)
        #   value (double precision)
        # [+ tb_idata_values_detail_hist]
        #   value_type (varchar(256))
        #   value_insert_time (varchar(256))
        #   area_id_fk (bigint)
        # -------------------------------------------------------------------

        self.iDataValuesMaster \
            = Table('tb_idata_values_master', self.metadata, autoload_with=self.engine)
        # [tb_idata_values_master]
        # id (bigint)
        # start_date (date)
        # end_date (date)
        # name (varchar(256))
        # created_date (timestamp(6))
        # idata_master_fk (bigint)
        # loan_updated_time (timestamp)
        # published (char(1))
        # isscenario (char(1))
        # temp_ind (char(1))
        # area_id (bigint)
        # last_updated_date (timestamp)
        # published_id (bigint)
        # note (text)

        self.iDataValuesDetail \
            = Table('tb_idata_values_detail', self.metadata, autoload_with=self.engine)
        # [tb_idata_values_detail]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)

        self.iDataValuesDetailHist \
            = Table('tb_idata_values_detail_hist', self.metadata, autoload_with=self.engine)
        # [tb_idata_values_detail_hist]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)
        # value_type (varchar(256))
        # value_insert_time (varchar(256))
        # area_id_fk (bigint)

        # -------------------------------------------------------------------

        return

    def _exists_id(self, what: Union[int, str], table: Table, columns: list[str], idcol: str = "id") -> bool:
        # with self.engine.connect() as conn:
        #     if isinstance(what, int):
        #         query = select(func.count()).where(table.c[idcol] == what)
        #         self.log.debug(query)
        #         count = conn.execute(query).scalar()
        #         return count > 0
        #     for col in columns:
        #         query = select(func.count()).where(table.c[col] == what)
        #         self.log.debug(query)
        #         count = conn.execute(query).scalar()
        #         if count > 0:
        #             return True
        #         else:
        #             continue
        # return False
        return self._convert_id(what, table, columns, idcol, nullable=True) is not None

    def _convert_id(self, what: Union[int, str], table: Table, columns: list[str], idcol: str = "id",
                    nullable=False) -> Optional[int]:
        """
        Convert a string into an id

        :param what: string to convert
        :param table: table to use
        :param columns: list of columns where to search the text
        :param idcol: column containing the 'id' value
        :return: the id as integer value
        """
        # check if 'what' is an integer or an integer in string format
        try:
            id = int(what)
            return id
        except:
            pass

        # 'what' is a string. Check a record with this string in one of the selected columns
        with self.engine.connect() as conn:
            for col in columns:
                query = select(table.c[idcol]).where(table.c[col] == what)
                result = conn.execute(query).fetchall()
                if len(result) > 0:
                    # [(id,)]
                    return result[0][0]
                continue
        if nullable:
            return None
        raise ValueError(f"Unable to convert '{what}' into an id using {table.name}")

    def _convert_name(self, id: str, table: Table, namecol: str, idcol: str = 'id') -> str:
        """

        :param id: id to convert
        :param table: table to use
        :param namecol: column containing the name
        :param idcol: column containing the 'id' value
        :return: the found name
        """
        with self.engine.connect() as conn:
            query = select(table.c[namecol]).where(table.c[idcol] == id)
            result = conn.execute(query).scalar()
            return result
        # raise ValueError(f"Unable to convert '{id}' into a name using {table.name}")
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
