import stdlib.loggingx as logging
import pandas as pd
from typing import Optional, Union, Any, Literal
from datetime import datetime

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


def concatenate_no_skill_df(df_with_skill, df_no_skill, skill_features_dict: dict[int, str]):
    # this code implements the same logic implemented in 'replicateUnskilledMeasuresAgainstAllSkilledMeasures(...)'
    if len(df_no_skill) == 0:
        return df_with_skill

    logging.getLogger('ipom.om').error("Function 'concatenate_no_skill_df(...)' Not implemented yet")
    return df_with_skill


def fill_missing_measures(df_pivoted: pd.DataFrame, measure_dict: dict[int, str], new_format: bool) -> pd.DataFrame:
    measure_ids = measure_dict.keys()
    # add missing columns (str(measure_id) OR measure_name) if necessary
    for mid in measure_ids:
        mname = measure_dict[mid] if new_format else str(mid)
        if mname not in df_pivoted.columns:
            df_pivoted[mname] = 0.
    return df_pivoted


def fill_missing_dates(df_pivoted, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    pass

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
    # end

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
    # end

    def __repr__(self):
        return f"{self.name}:{self.id}"
# end


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

    def feature_ids(self, name=False) -> Union[set[int], dict[int, str]]:
        with self.engine.connect() as conn:
            tdetail = self.ipom.AttributeDetail
            query = select(tdetail.c['id', 'attribute']).where(tdetail.c['attribute_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # rlist: [(id, name), ...]
            if name:
                return {rec[0]:rec[1] for rec in rlist}
            else:
                return {rec[0] for rec in rlist}

    def features(self) -> list[AttributeDetail]:
        return self.details()

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
# end


class PeriodHierarchy(IPlanObject):
    def __init__(self, ipom, period_hierarchy, period_length):
        super().__init__(ipom)
        self.period_hierarchy = period_hierarchy
        self.period_length = period_length
    # end

    def __repr__(self):
        return f"{self.period_hierarchy}:{self.period_length}"
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


class IDataModelMaster(IPlanData):
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
    def data_model(self) -> IDataModelMaster:
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
        pass

    @property
    def name(self):
        return self.data['ipr_conf_master_name']

    @property
    def description(self):
        return self.data['ipr_conf_master_desc']

    @property
    def data_model(self):
        data_model_id = self.data['idata_model_details_id_fk']
        return self.ipom.data_model(data_model_id)

    @property
    def area_hierarchy(self):
        area_hierarchy_id = self.data['area_id_fk']
        return self.ipom.area_hierarchy(area_hierarchy_id)

    @property
    def skill_hierarchy(self):
        skill_hierarchy_id = self.data['skill_id_fk']
        return self.ipom.skill_hierarchy(skill_hierarchy_id)

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
            query = select(tdetail.c['parameter_id', 'parameter_value']).where(tdetail.c['ipr_conf_master_id'] == self.id)
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

    @property
    def measure_ids(self) -> list[int]:
        input_ids, target_ids = self.input_target_measure_ids
        return input_ids + target_ids

    # Note: the COLUMN 'tb_ipr_conf_master_focussed.idata_id_fk' IS NULL
    #       in ALL records in the table!
    #       HOWEVER, in "theory" it is possible to retrieve teh 'Data Master' based on the
    #       values:  (data_model_id, area_hierarchy_id, skill_hierarchy_id)

    # @property
    # def data_master(self) -> Optional[IDataMaster]:
    #     data_master_id = self.data['idata_id_fk']
    #     return self.ipom.data_master(data_master_id) if data_master_id is not None else None

    @property
    def data_master(self) -> Optional[IDataMaster]:
        if self._data_master is None:
            data_model_id = self.data['idata_model_details_id_fk']
            area_hierarchy_id = self.data['area_id_fk']
            skill_hierarchy_id = self.data['skill_id_fk']
            self._data_master = self.ipom.find_data_master(data_model_id, area_hierarchy_id, skill_hierarchy_id)
        return self._data_master

    def details(self) -> list[IPredictDetailFocussed]:
        with self.engine.connect() as conn:
            tdetail = self.ipom.iPredictDetailFocussed
            query = select(tdetail).where(tdetail.c['ipr_conf_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        return [IPredictDetailFocussed(self.ipom, to_data(result), tdetail) for result in rlist]

    def select_training_data(self, new_format=False) -> pd.DataFrame:
        data_model_id = self.data['idata_model_details_id_fk']
        area_hierarchy_id = self.data['area_id_fk']
        skill_hierarchy_id = self.data['skill_id_fk']
        measure_ids = self.measure_ids
        # input_target_measure_ids = self.input_target_measure_ids

        return self.ipom.select_training_data(data_model_id, area_hierarchy_id, skill_hierarchy_id, measure_ids, new_format=new_format)

    def select_prediction_data(self, plan_id: Optional[int]=None, new_format=False) -> pd.DataFrame:
        data_model_id = self.data['idata_model_details_id_fk']
        area_hierarchy_id = self.data['area_id_fk']
        skill_hierarchy_id = self.data['skill_id_fk']
        input_target_measure_ids = self.input_target_measure_ids

        return self.ipom.select_prediction_data(
            data_model_id, area_hierarchy_id, skill_hierarchy_id,
            input_target_measure_ids,
            plan_id=plan_id,
            new_format=new_format)
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
# IPlanObjectModel
# ---------------------------------------------------------------------------

class IPlanObjectModel(IPlanObject):

    def __init__(self, url: Union[str, URL], **kwargs):
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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        pass

    # -----------------------------------------------------------------------

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

    def data_model(self, id: Union[int, str]) -> IDataModelMaster:
        if isinstance(id, str):
            id = self._convert_id(id, self.iDataModelMaster, ['description'])
        return IDataModelMaster(self, id, self.iDataModelMaster)

    def data_master(self, id: Union[int, str]) -> IDataMaster:
        if isinstance(id, str):
            id = self._convert_id(id, self.iDataMaster, ['description'])
        return IDataMaster(self, id, self.iDataMaster)

    def find_data_master(self, data_model_id: int, area_hierarchy_id: int, skill_hierarchy_id: int) -> Optional[IDataMaster]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataMaster
            query = select(table).where((table.c['area_id_fk'] == area_hierarchy_id) &
                                        (table.c['skill_id_fk'] == skill_hierarchy_id) &
                                        (table.c['idatamodel_id_fk'] == data_model_id))
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            if len(rlist) == 1:
                return IDataMaster(self.ipom, to_data(rlist[0]), table)
            elif len(rlist) == 0:
                return None
            else:
                raise ValueError(f"Multiple DataMasters with ({data_model_id}, {area_hierarchy_id}, {skill_hierarchy_id})")

    # -----------------------------------------------------------------------

    def data_values_master(self, id: Union[int, str]) -> IDataValuesMaster:
        if isinstance(id, str):
            id = self._convert_id(id, self.iDataValuesMaster, ['name'])
        return IDataValuesMaster(self, id, self.iDataValuesMaster)

    def select_data_values(self, data_values_master_id) -> pd.DataFrame:
        with self.engine.connect() as conn:
            table = self.iDataValuesDetail
            query = select(table.c['state_date', 'model_detail_id_fk', 'skill_id_fk', 'value']) \
                .where(table.c['value_master_fk'] == data_values_master_id)
            self.log.debug(query)
            df = pd.read_sql_query(query, self.engine)
        return df

    # -----------------------------------------------------------------------

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
    # end

    # -----------------------------------------------------------------------

    def predict_focussed(self, id: Union[int, str]) -> IPredictMasterFocussed:
        if isinstance(id, str):
            id = self._convert_id(id, self.iPredictMasterFocussed, ['ipr_conf_master_name', 'ipr_conf_master_desc'])
        return IPredictMasterFocussed(self, id, self.iPredictMasterFocussed)

    # -----------------------------------------------------------------------
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

    def select_training_data(self, data_model_id: int,
                             area_hierarchy_id: int,
                             skill_hierarchy_id: int,
                             measure_ids: list[int],
                             new_format=False) -> pd.DataFrame:
        """
        Retrieve the historical data from 'tb_idata_values_detail_hist' based on

            - data model id
            - area hierarchy id
            - skill hierarchy id
            - list of measure ids

        It is possible to replace the area/skill/measure ids with the correspondent names

        :param data_model_id: Data Model id
        :param area_hierarchy_id: Area Hierarchy id
        :param skill_hierarchy_id: Skill Hierarchy id
        :param measure_ids: list of selected measure ids
        :param new_format: if to create a dataframe compatible with
                the current implementation of the new format

        :return: a dataframe with the following columns
                if 'new_format == True':
                    columns: ['area:str', 'skill:str', 'date:datetime', <measure_1:float>, ...]
                else
                    columns: ['skill_id_fk:int', 'area_id_fk:int', 'time:datetime', 'day:str', <measure_1: float>, ...]
        """

        table = self.iDataValuesDetailHist

        # 1) retrieve all area/skill feature ids
        area_feature_dict = self.area_hierarchy(area_hierarchy_id).feature_ids(name=True)
        area_feature_ids = list(area_feature_dict.keys())

        skill_feature_dict = self.skill_hierarchy(skill_hierarchy_id).feature_ids(name=True)
        skill_feature_ids = list(skill_feature_dict.keys())

        measure_dict: dict[int, str] = self.select_measure_names(measure_ids)

        # 2) retrieve the data with 'skill NOT NULL'
        query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
            .where(table.c['model_detail_id_fk'].in_(measure_ids) &
                   table.c['area_id_fk'].in_(area_feature_ids) &
                   table.c['skill_id_fk'].in_(skill_feature_ids))
        self.log.debug(query)
        # rlist = conn.execute(query).fetchall()
        df_with_skill = pd.read_sql_query(query, self.engine)

        # 3) retrieve the data with 'skill IS NULL'
        query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
            .where(table.c['model_detail_id_fk'].in_(measure_ids) &
                   table.c['area_id_fk'].in_(area_feature_ids) &
                   (table.c['skill_id_fk'] == None))
        self.log.debug(query)
        df_no_skill = pd.read_sql_query(query, self.engine)

        # 4) concatenate df_with_skill WITH df_no_skill
        df = concatenate_no_skill_df(df_with_skill, df_no_skill, skill_feature_dict)

        return self._pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format)

    def _pivot_df(self, df: pd.DataFrame,
                  area_feature_dict: dict[int, str], skill_feature_dict: [int, str], measure_dict: [int, str],
                  new_format: bool) -> pd.DataFrame:

        # 5) replace area/skill ids with the names
        if new_format:
            df.replace(to_replace={
                'area_id_fk': area_feature_dict,
                'skill_id_fk': skill_feature_dict,
                'model_detail_id_fk': measure_dict
            }, inplace=True)

        # 6) transpose df & move the multiindex as columns
        #    it creates a pandas DataFrame with the following columns:
        #
        #       columns=['skill_id_fk', 'area_id_fk', 'time', 'day',' <measure_id_1>', ...]
        #
        # However this structure is not 'human readable'. Then, the new format is
        #
        #       columns=['area', 'skill', 'date', '<measure_name_1>', ...]
        #
        # with 'area' & 'skill' values as strings and the measure column names the measure name
        #
        index_columns = ['state_date', 'skill_id_fk', 'area_id_fk']
        date_column = 'state_date'

        df_pivoted = df.pivot_table(
            index=index_columns,
            columns=['model_detail_id_fk'],
            values='value'
        ).fillna(0)

        df_pivoted.reset_index(inplace=True, names=index_columns)
        df_pivoted[date_column] = pd.to_datetime(df_pivoted[date_column])

        if new_format:
            df_pivoted.rename(columns={
                'area_id_fk': 'area',
                'skill_id_fk': 'skill',
                'state_date': 'date',
            }, inplace=True)
        else:
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
        # end

        return df_pivoted
    # end

    def select_prediction_data(
        self, data_model_id: int,
        area_hierarchy_id: int, skill_hierarchy_id: int,
        input_target_measure_ids: tuple[list[int], list[int]],
        plan_id: Optional[int] = None,
        new_format=False) -> pd.DataFrame:

        input_measure_ids, target_measure_ids = input_target_measure_ids
        measure_ids: list[int] = input_measure_ids + target_measure_ids

        area_feature_dict = self.area_hierarchy(area_hierarchy_id).feature_ids(name=True)
        area_feature_ids = list(area_feature_dict.keys())

        skill_feature_dict = self.skill_hierarchy(skill_hierarchy_id).feature_ids(name=True)
        skill_feature_ids = list(skill_feature_dict.keys())

        measure_dict: dict[int, str] = self.select_measure_names(measure_ids)

        # Note: [tb_idata_values_detail]
        #   DOESNT' CONTAIN 'area_id_fk'
        #   BUT it has a reference with [tb_idata_values_master] ('value_master_fk)
        #   AND 'tb_idata_values_master' contains 'area_id', that is, the required 'area_id_fk'
        #
        # To select the data, it is necessary to implement a complex trick!

        if plan_id is None:
            data_values_master_ids, start_date, end_date = self._select_data_values_masters_ids(
                data_model_id, area_hierarchy_id, skill_hierarchy_id)
        else:
            data_values_master = self.data_values_master(plan_id)
            start_date = data_values_master.start_date
            end_date = data_values_master.end_date
            data_values_master_ids = [plan_id]

        # 3) retrieve all data_values_detail having
        #       value_master_fk    in data_values_master_ids
        #       model_detail_id_fk in measure_ids
        #       skill_id_fk        in skill_feature_ids
        #
        # WARN: 'tb_idata_values_detail' DOESN'T contains 'area_id_fk'. It is necessary to do a join with
        #       'tb_idata_values_master' containing 'area_id' that can use as 'area_id_fk'
        #
        df = self._select_data_values_details(data_values_master_ids, skill_feature_ids, measure_ids)

        # 4) pivot the table
        df_pivoted = self._pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format)

        # 5) fill the missing values
        df_pivoted = fill_missing_measures(df_pivoted, measure_dict, new_format)
        df_pivoted = fill_missing_dates(df_pivoted, start_date, end_date)

        # done
        return df_pivoted
    # end

    def _select_data_values_masters_ids(self, data_model_id, area_hierarchy_id, skill_hierarchy_id) \
            -> tuple[set[id], datetime, datetime]:
        area_feature_dict = self.area_hierarchy(area_hierarchy_id).feature_ids(name=True)
        area_feature_ids = list(area_feature_dict.keys())

        with self.engine.connect() as conn:
            # 1) retrieve all data_master_id having
            #       idatamodel_id_fk == data_model_id
            #       area_id_fk       == area_hierarchy_id
            #       skill_id_fk      == skill_hierarchy_id
            table = self.iDataMaster
            query = select(table.c.id).where(
                (table.c['idatamodel_id_fk'] == data_model_id) &
                (table.c['area_id_fk']       == area_hierarchy_id) &
                (table.c['skill_id_fk']      == skill_hierarchy_id)
            )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            data_master_ids = {result[0] for result in rlist}

            # 2) retrieve all data_value_master_id having
            #       idata_master_fk in data_master_ids
            #       area_id         in area_feature_ids
            table = self.iDataValuesMaster
            query = select(table.c.id).where(
                (table.c['idata_master_fk'].in_(data_master_ids)) &
                (table.c['area_id'].in_(area_feature_ids))
            )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            data_values_master_ids = {result[0] for result in rlist}

            # 3) retrieve start_date, end_date
            table = self.iDataValuesMaster
            query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
                (table.c['idata_master_fk'].in_(data_master_ids)) &
                (table.c['area_id'].in_(area_feature_ids))
            )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            if len(rlist) > 0:
                start_date, end_date = rlist[0]
            else:
                start_date, end_date = None, None
        # end
        return data_values_master_ids, start_date, end_date
    # end

    def _select_data_values_details(self, data_values_master_ids, skill_feature_ids, measure_ids) \
            -> pd.DataFrame:
        qtext = """
        select tivm.area_id as area_id_fk, tivd.skill_id_fk as skill_id_fk, tivd.model_detail_id_fk as model_detail_id_fk, 
               tivd.state_date as state_date, tivd.value as value
         from tb_idata_values_detail as tivd, tb_idata_values_master as tivm
        where tivd.value_master_fk in :value_master_fk
          and tivd.model_detail_id_fk in :model_detail_id_fk
          and tivd.skill_id_fk in :skill_id_fk
          and tivm.id = tivd.value_master_fk
        """
        query = text(qtext)
        self.log.debug(query)
        df = pd.read_sql_query(query, self.engine, params=dict(
            value_master_fk=tuple(data_values_master_ids),
            model_detail_id_fk=tuple(measure_ids),
            skill_id_fk=tuple(skill_feature_ids)
        ))
        return df
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
    # end

    def _convert_id(self, what: str, table: Table, columns: list[str], idcol: str = "id") -> int:
        """

        :param what: string to search
        :param table: table to use
        :param idcol: columns containind the 'id' value
        :param columns: list of columns where to serach the text
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
    # end

    # -----------------------------------------------------------------------
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
