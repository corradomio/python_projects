import stdlib.loggingx as logging
import traceback
from deprecated import deprecated
from stdlib import as_list
from stdlib.is_instance import is_instance
from stdlib.dateutilx import relativeperiods
from stdlib.dict import reverse
import pandas as pd
import pandasx as pdx
from pandas import DataFrame
from typing import Optional, Union, Any, Literal
from datetime import datetime, timedelta

from sqlalchemy import MetaData, Engine, Table, Row, create_engine, URL, bindparam
from sqlalchemy import select, delete, insert, text, func


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
    except:
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


def compose_predict_df(df: DataFrame,
                       area_feature_dict: dict[int, str],
                       skill_feature_dict: dict[int, str],
                       input_measure_ids: list[int],
                       measure_dict: dict[int, str],
                       start_end_dates: dict[int, tuple[datetime, datetime]],
                       freq: Literal['D', 'W', 'M'],
                       defval: Union[None, float] = 0.,
                       new_format=False) -> DataFrame:
    assert is_instance(defval, Union[None, float])

    if new_format:
        return _compose_predict_df_new_format(
            df,
            area_feature_dict,
            skill_feature_dict,
            input_measure_ids,
            measure_dict,
            start_end_dates, freq,
            defval)
    else:
        return _compose_predict_df_old_format(
            df,
            area_feature_dict,
            skill_feature_dict,
            input_measure_ids,
            measure_dict,
            start_end_dates, freq,
            defval)
    # end


def _compose_predict_df_old_format(
    df: DataFrame,
    area_feature_dict: dict[int, str],
    skill_feature_dict: dict[int, str],
    input_measure_ids: list[int],
    measure_dict: dict[int, str],
    start_end_dates: dict[int, tuple[datetime, datetime]],
    freq: Literal['D', 'W', 'M'],
    defval: Union[None, float] = 0.) -> DataFrame:

    # columns: ['skill_id_fk', 'area_id_fk', 'time', 'day', <measure_id>

    target_measure_ids = set(measure_dict.keys()).difference(input_measure_ids)

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
            for target_id in target_measure_ids:
                area_skill_df[str(target_id)] = defval

            df_list.append(area_skill_df)

    df = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)
    return df


def _compose_predict_df_new_format(
    df: DataFrame,
    area_feature_dict: dict[int, str],
    skill_feature_dict: dict[int, str],
    input_measure_ids: list[int],
    measure_dict: dict[int, str],
    start_end_dates: dict[int, tuple[datetime, datetime]],
    freq: Literal['D', 'W', 'M'],
    defval: Union[None, float] = 0.) -> DataFrame:

    # columns: ['area', 'skill', 'date', <measure_name>]

    target_measure_ids = set(measure_dict.keys()).difference(input_measure_ids)

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
                'area': area_feature_dict[area_feature_id],
                'skill': skill_feature_dict[skill_feature_id],
                'date': date_index.to_series()
            })
            for target_id in target_measure_ids:
                target_name = measure_dict[target_id]
                area_skill_df[target_name] = defval

            df_list.append(area_skill_df)

    df = pd.concat(df_list, axis=0, ignore_index=True).reset_index(drop=True)
    return df

    pass


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

    new_format = False
    measure_ids = list(measure_dict.keys())

    area_features_drev = reverse(area_features_dict)
    skill_feature_drev = reverse(skill_features_dict)
    measure_drev = reverse(measure_dict)

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

    def feature_ids(self, with_name=False) -> Union[list[int], dict[int, str]]:
        with self.engine.connect() as conn:
            tdetail = self.ipom.AttributeDetail
            query = select(tdetail.c['id', 'attribute']).where(tdetail.c['attribute_master_id'] == self.id)
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

    def to_ids(self, attr: Union[None, int, list[int], str, list[str]]):
        """
        Convert the attribute(s) in a list of attribute ids.
        The attribute can be specified as id (an integer) or name (a string)
        :param attr: attribute(s) to convert
        :return: list of attribute ids
        """
        feature_dict = self.feature_ids(with_name=True)
        feature_drev = reverse(feature_dict)
        aids = []

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
            tmeasure = self.ipom.iDataModelDetail
            query = select(tmeasure).where(tmeasure.c['data_model_id_fk'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        return [Measure(self.ipom, to_data(result), tmeasure) for result in rlist]

    def measures(self) -> list[Measure]:
        return self.details()


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


# ---------------------------------------------------------------------------
# IDataValuesMaster == IPredictionPlans
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

    def exists_plan(self, name_or_date: Union[str, datetime], data_master_id: int):
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
                    force=False
                    ):
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

        # retrieve the Data Master
        if isinstance(data_master, int | str):
            data_master = self.ipom.data_master(data_master)
        data_master_id = data_master.id

        if name is None:
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            name = f"Auto_Plan_OM_{now_str}"

        already_exists = self.exists_plan(name, data_master_id)
        if already_exists and not force:
            return

        if already_exists:
            self.delete_plan(name)

        af_ids: list[int] = data_master.area_hierarchy.feature_ids()
        if len(area_feature_ids) == 0:
            area_feature_ids = af_ids
        else:
            af_count = len(area_feature_ids)
            ai_count = len(af_ids.intersection(area_feature_ids))
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
                    note="created by Python IPlanObjectModel"
                )
                if count == 0: self.log.debug(stmt)
                conn.execute(stmt)
                count += 1
            conn.commit()
        # end
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

    def exists(self):
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

    def create(self, start_date: datetime, end_date: Optional[datetime] = None, periods: Optional[int] = None,
               force=False, note=None):
        """
        Create a plan with the specified name for all areas in the area hierarchy
        If end_date or periods are not specified, it is used the PeriodHierarchy of the DataMaster

        :param start_date: start date
        :param end_date: end date
        :param periods: n of periods
        :param force: if to recreate
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
        if already_exists and not force:
            self.log.warning(f"Plan {name} already existent")
            return

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
        note = "created by Python IPlanObjectModel" if note is None else note
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
                )
                if count == 0: self.log.debug(stmt)
                conn.execute(stmt)
                count += 1
            conn.commit()
        # end
        self.log.info(f"Done")
        return
    # end
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



class IDataValue(IPlanData):
    def __init__(self, ipom, id, table):
        super().__init__(ipom, id, table)


# ---------------------------------------------------------------------------
# IPredictTimeSeries
# ---------------------------------------------------------------------------

class IPredictTimeSeries(IPlanObject):

    def __init__(self, ipom,
                 id: int,
                 data_master_id: int,
                 data_values_master_ids: list[int]):
        super().__init__(ipom)

        assert is_instance(id, int)
        assert is_instance(data_master_id, int)
        assert is_instance(data_values_master_ids, list[int])

        self._id = id
        self._data_master_id = data_master_id

        # data_values_master_ids are compatible with 'Data Master' and 'Area Feature' ids
        # FOR CONSTRUCTION
        self._data_values_master_ids = data_values_master_ids

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
    # Train/predict data
    # -----------------------------------------------------------------------

    def select_train_data(self,
                          area: Union[None, int, list[int]] = None,
                          skill: Union[None, int, list[int]] = None,
                          new_format=False) \
            -> DataFrame:

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])

        area_ids: list[int] = self.area_hierarchy.to_ids(area)
        skill_ids: list[int] = self.skill_hierarchy.to_ids(skill)
        # Note: 'data_values_master_ids' automatically specify 'data_master_id'

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)

        df: DataFrame = self.ipom.select_train_data(
            self._data_master_id,
            self._data_values_master_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            new_format=new_format
        )
        return df

    def select_predict_data(self,
                            start_date: Optional[datetime] = None,
                            periods: Optional[int] = None,
                            area: Union[None, int, list[int], str, list[str]] = None,
                            skill: Union[None, int, list[int], str, list[str]] = None,
                            new_format=False) \
            -> DataFrame:
        """

        :param start_date: optional start date
        :param periods: n of periods. The period depends on the DataModel::PeriodHierarchy
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param new_format:
        :return: the dataframe containing the past and the future data
        """

        assert is_instance(start_date, Optional[datetime])
        assert is_instance(periods, Optional[int])
        assert is_instance(area, Union[None, int, list[int]])
        assert is_instance(skill, Union[None, int, list[int]])

        end_date = None
        area_ids: list[int] = self.area_hierarchy.to_ids(area)
        skill_ids: list[int] = self.skill_hierarchy.to_ids(skill)
        # Note: 'data_values_master_ids' automatically specify 'data_master_id'

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        input_feature_ids, _ = self.input_target_measure_ids

        freq = self.period_hierarchy.freq
        if periods is None:
            periods = self.period_hierarchy.periods
        if start_date is not None:
            end_date = start_date + relativeperiods(periods=periods, freq=freq)

        df: DataFrame = self.ipom.select_predict_data(
            self._data_master_id,
            self._data_values_master_ids,
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
                - False: the data in the dataset replaces the same data in the database
                        (update or insert)
                - True:  all data in the database is not deleted or updated
                        (insert only)
        """
        assert is_instance(df, DataFrame)
        assert is_instance(plan, Optional[str])

        self.log.info("Save train data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)

        if plan is not None:
            pplan = self.ipom.prediction_plan(plan, self.data_master.id)
            area_plan_map = pplan.area_plan_map()
        else:
            area_plan_map = None

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
                    self.log.debugt(f"... area:{area_name}, skill:{skill_name}, measure:{measure_name}")

                    self.ipom.save_area_skill_train_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id),
                        dfas, plan_id=plan_id, update=update)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to create plan for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return

    def delete_train_data(self,
                          plan: str,
                          start_date: Optional[datetime] = None,
                          periods: Optional[int] = None,
                          area: Union[None, int, list[int], str, list[str]] = None,
                          skill: Union[None, int, list[int], str, list[str]] = None,):

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])

        area_ids: list[int] = self.area_hierarchy.to_ids(area)
        skill_ids: list[int] = self.skill_hierarchy.to_ids(skill)
        # Note: 'data_values_master_ids' automatically specify 'data_master_id'

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)

        pplan = self.ipom.prediction_plan(plan, self.data_master.id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        self.ipom.delete_train_data(
            plan,
            self._data_master_id,
            self._data_values_master_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
        )

    def save_predict_data(self, df: DataFrame, plan: str, update: Optional[bool] = None):
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
                - None:  all data is deleted and replaced
                - False: the data in the dataset replaces the same data in the database
                         (or it is inserted)
                - True:  all data in the database is not deleted or updated
        """
        assert is_instance(df, DataFrame)
        assert is_instance(plan, str)

        self.log.info("Save prediction data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)

        # 0) retrieve the Plan map
        #
        pplan = self.ipom.prediction_plan(plan, self.data_master.id)
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
                    self.log.debugt(f"... area:{area_name}, skill:{skill_name}, measure:{measure_name}")

                    self.ipom.save_area_skill_predict_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id),
                        dfas, plan_id=int(plan_id), update=update)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to create plan for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return

    def delete_predict_data(self,
                            area: Union[None, int, list[int]] = None,
                            skill: Union[None, int, list[int]] = None,):
        pass
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

    def hierarchy_type(self, id: Union[int, str]) -> Literal["area", "skill"]:
        if isinstance(id, str):
            id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])

        with self.engine.connect() as conn:
            table = self.AttributeMaster
            query = select(table.c['hierarchy_type']).where(table.c['id'] == id)
            self.log.debug(f"{query}")
            hierarchy_type = conn.execute(query).fetchone()[0]
        return "area" if hierarchy_type == 1 else "skill"

    def area_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        if isinstance(id, str):
            id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        ah = AttributeHierarchy(self.ipom, id, self.AttributeMaster)
        assert ah.type == "area"
        return ah

    def area_feature(self, id: Union[int, str]) -> AttributeDetail:
        if isinstance(id, str):
            id = self._convert_id(id, self.AttributeDetail, ['attribute', 'description'])
        af = AttributeDetail(self.ipom, id, self.AttributeDetail)
        assert self.hierarchy_type(af.hierarchy_id) == "area"
        return af

    def skill_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
        if isinstance(id, str):
            id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        sk = AttributeHierarchy(self.ipom, id, self.AttributeMaster)
        assert sk.type == "skill"
        return sk

    def skill_feature(self, id: Union[int, str]) -> AttributeDetail:
        if isinstance(id, str):
            id = self._convert_id(id, self.AttributeDetail, ['attribute', 'description'])
        sf = AttributeDetail(self.ipom, id, self.AttributeDetail)
        assert self.hierarchy_type(sf.hierarchy_id) == "skill"
        return sf

    # -----------------------------------------------------------------------
    # Data Model
    # Data Master

    def data_model(self, id: Union[int, str]) -> IDataModel:
        if isinstance(id, str):
            id = self._convert_id(id, self.iDataModelMaster, ['description'])
        return IDataModel(self, id, self.iDataModelMaster)

    def data_master(self, id: Union[int, str]) -> IDataMaster:
        if isinstance(id, str):
            id = self._convert_id(id, self.iDataMaster, ['description'])
        return IDataMaster(self, id, self.iDataMaster)

    def find_data_master(self, data_model_id: int, area_hierarchy_id: int, skill_hierarchy_id: int) \
            -> Optional[IDataMaster]:

        with self.engine.connect() as conn:
            table = self.ipom.iDataMaster
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

    def data_values_master(self, id: Union[int, str]) -> IDataValuesMaster:
        if isinstance(id, str):
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

    def select_data_values_master_ids(
        self,
        name: Optional[str],
        data_master_ids: list[int],
        area_feature_ids: list[int],
    ) -> list[int]:
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

    def select_data_values_master_date_interval(self, data_values_master_ids: list[int]) -> tuple[datetime, datetime]:
        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                table.c.id.in_(data_values_master_ids)
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
            data_values_master_ids = self._select_data_values_master_ids([data_master_id], area_feature_ids)
        else:
            data_values_master_ids, data_master_id = self._select_data_values_master_ids_by_plan(plan, area_feature_ids)

        # data_values_master_ids are compatible with Data Master and Area Feature ids FOR CONSTRUCTION
        return IPredictTimeSeries(self, pmf.id, data_master_id, data_values_master_ids)

    def predict_master_focussed(self, id: Union[int, str]) -> IPredictMasterFocussed:
        if isinstance(id, str):
            id = self._convert_id(id, self.iPredictMasterFocussed, ['ipr_conf_master_name', 'ipr_conf_master_desc'])
        return IPredictMasterFocussed(self, id, self.iPredictMasterFocussed)

    # -----------------------------------------------------------------------
    # data_values_detail_hist
    # data_values_detail

    def save_area_skill_train_data(self, area_feature_id: int, skill_feature_id: int, measure_id: int, df: DataFrame,
                                   plan_id: Optional[int] = None, update: Optional[bool] = None):
        assert is_instance(area_feature_id, int)
        assert is_instance(skill_feature_id, int)
        assert is_instance(measure_id, int)
        assert is_instance(df, DataFrame)
        assert is_instance(plan_id, Optional[int])

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

                query = """
                    DELETE FROM tb_idata_values_detail_hist
                     WHERE state_date >= :start_date
                       AND state_date <= :end_date
                       AND model_detail_id_fk = :measure_id
                       AND area_id_fk = :area_feature_id
                       AND skill_id_fk = :skill_feature_id
                """
                try:
                    conn.execute(text(query), parameters=dict(
                        start_date=start_date,
                        end_date=end_date,
                        measure_id=measure_id,
                        area_feature_id=area_feature_id,
                        skill_feature_id=skill_feature_id
                    ))
                except Exception as e:
                    pass

            # TODO: BETTER IMPLEMENTATION with 'prepared_statement'
            for i in range(n):
                state_date = pdx.to_datetime(df['state_date'].iloc[i])
                value = float(df[measure_id].iloc[i])
                query = insert(table).values(
                    area_id_fk=area_feature_id,
                    skill_id_fk=skill_feature_id,
                    model_detail_id_fk=measure_id,
                    state_date=state_date,
                    value=value,

                    value_master_fk=plan_id,
                    updated_date=now,
                    value_type=None,
                    value_insert_time=None,
                )
                # self.log.debug(query)
                conn.execute(query)
            conn.commit()
        return

    def save_area_skill_predict_data(self, area_feature_id: int, skill_feature_id: int, measure_id: int, df: DataFrame,
                                     plan_id: int, update: Optional[bool] = None):
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

                query = """
                    DELETE FROM tb_idata_values_detail
                     WHERE state_date >= :start_date
                       AND state_date <= :end_date
                       AND model_detail_id_fk = :measure_id
                       AND skill_id_fk = :skill_feature_id
                """
                try:
                    conn.execute(text(query), parameters=dict(
                        start_date=start_date,
                        end_date=end_date,
                        measure_id=measure_id,
                        area_feature_id=area_feature_id,
                        skill_feature_id=skill_feature_id
                    ))
                except Exception as e:
                    pass

            # TODO: BETTER IMPLEMENTATION with 'prepared_statement'
            for i in range(n):
                state_date = pdx.to_datetime(df['state_date'].iloc[i])
                value = float(df[measure_id].iloc[i])
                query = insert(table).values(
                    value_master_fk=plan_id,
                    state_date=state_date,
                    skill_id_fk=skill_feature_id,
                    model_detail_id_fk=measure_id,
                    value=value,
                    updated_date=now,
                )
                # self.log.debug(query)
                conn.execute(query)
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

    def select_train_data(self,
                          data_master_id: int,
                          data_values_master_ids: list[int],
                          area_feature_dict: dict[int, str],
                          skill_feature_dict: dict[int, str],
                          measure_dict: dict[int, str],
                          new_format=False) -> DataFrame:
        """
        Retrieve the historical data from 'tb_idata_values_detail_hist' based on

            - data_master_id
            - data_values_master_ids
            - area_feature_ids
            - skill_feature_ids
            - measure_ids

        It is possible to replace the area/skill/measure ids with the correspondent names

        :param data_master_id:
        :param data_values_master_ids:
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
        assert is_instance(data_values_master_ids, list[int])
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

        return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format)

    def delete_train_data(self,
                          data_master_id: int,
                          data_values_master_ids: list[int],
                          area_feature_dict: dict[int, str],
                          skill_feature_dict: dict[int, str],
                          measure_dict: dict[int, str]):

        assert is_instance(data_master_id, int)
        assert is_instance(data_values_master_ids, list[int])
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
                .where(table.c['model_detail_id_fk'].in_(measure_ids) &
                       table.c['area_id_fk'].in_(area_feature_ids) &
                       table.c['skill_id_fk'].in_(skill_feature_ids))
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return

    def select_predict_data(self,
                            data_master_id: int,
                            data_values_master_ids: list[int],
                            area_feature_dict: dict[int, str],
                            skill_feature_dict: dict[int, str],
                            input_measure_ids: list[int],
                            measure_dict: dict[int, str],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            freq: Literal['D', 'W', 'M'] = 'D',
                            new_format=False) -> DataFrame:

        assert is_instance(data_master_id, int)
        assert is_instance(data_values_master_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(input_measure_ids, list[int])
        assert is_instance(measure_dict, dict[int, str])
        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(freq, Literal['D', 'W', 'M'])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        # 2) retrieve start/end dates for each area
        start_end_date_dict = self._select_start_end_date_dict(data_values_master_ids, area_feature_ids)

        # add the DEFAULT start/end date for the areas without a date range
        # Note: it is used 0 (ZERO) as key
        if start_date is not None and end_date is not None:
            start_end_date_dict[0] = (start_date, end_date)

        # Note: [tb_idata_values_detail]
        #   DOESNT' CONTAIN 'area_id_fk'
        #   BUT it has a reference with [tb_idata_values_master] ('value_master_fk)
        #   AND 'tb_idata_values_master' contains 'area_id', that is, the required 'area_id_fk'
        df = self._select_predict_data(data_values_master_ids, area_feature_ids, skill_feature_ids, measure_ids)

        df_pivoted = pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format)

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

    def delete_predict_data(self,
                            data_master_id: int,
                            data_values_master_ids: list[int],
                            area_feature_dict: dict[int, str],
                            skill_feature_dict: dict[int, str],
                            input_measure_ids: list[int],
                            measure_dict: dict[int, str],
                            start_date: Optional[datetime] = None,
                            end_date: Optional[datetime] = None,
                            freq: Literal['D', 'W', 'M'] = 'D',
                            new_format=False):

        assert is_instance(data_master_id, int)
        assert is_instance(data_values_master_ids, list[int])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(input_measure_ids, list[int])
        assert is_instance(measure_dict, dict[int, str])
        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(freq, Literal['D', 'W', 'M'])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        with self.engine.connect() as conn:
            qtext = """
                    delete
                     from tb_idata_values_detail as tivd, tb_idata_values_master as tivm
                    where tivd.value_master_fk in :data_values_master_ids
                      and tivd.model_detail_id_fk in :measure_ids
                      and tivd.skill_id_fk in :skill_feature_ids
                      and tivm.area_id in :area_feature_ids
                      and tivm.id in :data_values_master_ids
                      and tivm.id = tivd.value_master_fk
                    """
            query = text(qtext)
            self.log.debug(query)
            conn.execute(query, parameters=dict(
                data_values_master_ids=tuple(data_values_master_ids),
                measure_ids=tuple(measure_ids),
                skill_feature_ids=tuple(skill_feature_ids),
                area_feature_ids=tuple(area_feature_ids)
            ))
            conn.commit()
        return

    # -----------------------------------------------------------------------

    def _select_start_end_date_dict(self, data_values_master_ids, area_feature_ids) \
            -> dict[int, tuple[datetime, datetime]]:
        # 2) retrieve start/end dates for each area
        qtext = """
                select tivm.area_id as area_id_fk, tivm.start_date, tivm.end_date 
                  from tb_idata_values_master as tivm
                 where tivm.id in :data_values_master_ids 
                   and tivm.area_id in :area_feature_ids
                """
        query = text(qtext)
        self.log.debug(query)

        with self.engine.connect() as conn:
            rlist = conn.execute(query, parameters=dict(
                data_values_master_ids=tuple(data_values_master_ids),
                area_feature_ids=tuple(area_feature_ids)
            )).fetchall()
            start_end_date_dict = {
                r[0]: (r[1], r[2]) for r in rlist
            }
        return start_end_date_dict
    # end

    def _select_predict_data(self,
                             data_values_master_ids: list[int],
                             area_feature_ids: list[int],
                             skill_feature_ids: list[int],
                             measure_ids: list[int]) \
            -> DataFrame:
        qtext = """
                select tivm.area_id as area_id_fk, tivd.skill_id_fk as skill_id_fk, tivd.model_detail_id_fk as model_detail_id_fk, 
                       tivd.state_date as state_date, tivd.value as value
                 from tb_idata_values_detail as tivd, tb_idata_values_master as tivm
                where tivd.value_master_fk in :data_values_master_ids
                  and tivd.model_detail_id_fk in :measure_ids
                  and tivd.skill_id_fk in :skill_feature_ids
                  and tivm.area_id in :area_feature_ids
                  and tivm.id in :data_values_master_ids
                  and tivm.id = tivd.value_master_fk
                """
        query = text(qtext)
        self.log.debug(query)
        df = pd.read_sql_query(query, self.engine, params=dict(
            data_values_master_ids=tuple(data_values_master_ids),
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
            data_values_master_ids = list({result[0] for result in rlist})

        if len(data_master_ids) > 1:
            self.log.warning(f"Multiple Data Masters for plan {plan_name}")

        return data_values_master_ids, data_master_ids[-1]
    # end

    # def _select_data_values_masters_ids(self, data_model_id, area_hierarchy_id, skill_hierarchy_id) \
    #         -> tuple[set[id], datetime, datetime]:
    #     area_feature_dict = self.area_hierarchy(area_hierarchy_id).feature_ids(with_name=True)
    #     area_feature_ids = list(area_feature_dict.keys())
    #
    #     with self.engine.connect() as conn:
    #         # 1) retrieve all data_master_id having
    #         #       idatamodel_id_fk == data_model_id
    #         #       area_id_fk       == area_hierarchy_id
    #         #       skill_id_fk      == skill_hierarchy_id
    #         table = self.iDataMaster
    #         query = select(table.c.id).where(
    #             (table.c['idatamodel_id_fk'] == data_model_id) &
    #             (table.c['area_id_fk']       == area_hierarchy_id) &
    #             (table.c['skill_id_fk']      == skill_hierarchy_id)
    #         )
    #         self.log.debug(query)
    #         rlist = conn.execute(query).fetchall()
    #         data_master_ids = {result[0] for result in rlist}
    #
    #         # 2) retrieve all data_value_master_id having
    #         #       idata_master_fk in data_master_ids
    #         #       area_id         in area_feature_ids
    #         table = self.iDataValuesMaster
    #         query = select(table.c.id).where(
    #             (table.c['idata_master_fk'].in_(data_master_ids)) &
    #             (table.c['area_id'].in_(area_feature_ids))
    #         )
    #         self.log.debug(query)
    #         rlist = conn.execute(query).fetchall()
    #         data_values_master_ids = {result[0] for result in rlist}
    #
    #         # 3) retrieve start_date, end_date
    #         table = self.iDataValuesMaster
    #         query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
    #             (table.c['idata_master_fk'].in_(data_master_ids)) &
    #             (table.c['area_id'].in_(area_feature_ids))
    #         )
    #         self.log.debug(query)
    #         rlist = conn.execute(query).fetchall()
    #         if len(rlist) > 0:
    #             start_date, end_date = rlist[0]
    #         else:
    #             start_date, end_date = None, None
    #     # end
    #     return data_values_master_ids, start_date, end_date
    # # end

    # def _select_data_values_details(self, data_values_master_ids, skill_feature_ids, measure_ids) \
    #         -> DataFrame:
    #     qtext = """
    #     select tivm.area_id as area_id_fk, tivd.skill_id_fk as skill_id_fk, tivd.model_detail_id_fk as model_detail_id_fk,
    #            tivd.state_date as state_date, tivd.value as value
    #      from tb_idata_values_detail as tivd, tb_idata_values_master as tivm
    #     where tivd.value_master_fk in :value_master_fk
    #       and tivd.model_detail_id_fk in :model_detail_id_fk
    #       and tivd.skill_id_fk in :skill_id_fk
    #       and tivm.id = tivd.value_master_fk
    #     """
    #     query = text(qtext)
    #     self.log.debug(query)
    #     df = pd.read_sql_query(query, self.engine, params=dict(
    #         value_master_fk=tuple(data_values_master_ids),
    #         model_detail_id_fk=tuple(measure_ids),
    #         skill_id_fk=tuple(skill_feature_ids)
    #     ))
    #     return df
    # # end

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

    def _convert_id(self, what: str, table: Table, columns: list[str], idcol: str = "id") -> int:
        """
        Convert a string into an id

        :param what: string to convert
        :param table: table to use
        :param columns: list of columns where to search the text
        :param idcol: column containing the 'id' value
        :return: the id as integer value
        """
        # check if 'what' is an integer in string format
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
            result = conn.execute(query).fetchone()
            return result[0]
        raise ValueError(f"Unable to convert '{id}' into a name using {table.name}")
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
