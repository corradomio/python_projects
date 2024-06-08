from typing import Union, Optional
from stdlib import is_instance
import logging
from collections import deque
from sqlalchemy import MetaData, Engine, create_engine, URL, Table, ForeignKey


class DatabaseDAG:

    def __init__(self, url: Union[str, dict, URL], **kwargs):
        assert is_instance(url, Union[str, dict, URL])
        if is_instance(url, dict):
            datasource_dict: dict = url
            url = URL.create(**datasource_dict)

        self.engine: Optional[Engine] = None
        self.url = url
        self.metadata: Optional[MetaData] = None
        self.kwargs = kwargs
        self.log = logging.getLogger(f"dbdag.{self.__class__.__name__}")
        
    def connect(self, **kwargs) -> "DatabaseDAG":
        """
        It creates a 'connection' to the DBMS, NOT a connection to execute queries (a 'session')
        This step is used to create the data structures necessary to access the specific database
        (Python SQLAlchemy engine, metadata, ...)
        :param kwargs: passed to SQLAlchemy 'create_engine(...)' function
        """
        self.log.debug(f"connecting to {self.url}")

        self.engine = create_engine(self.url, **kwargs)
        self._load_metadata()

        self.log.info(f"connected")
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
    #
    def scan_all(self):

        processed = set()
        waiting = deque(list(self.metadata.tables.values()))

        index = 1
        while len(waiting) > 0:
            s = waiting.pop()
            if s in processed:
                continue
            processed.add(s)

            print(f"{index:3}: {s.name}")
            index += 1

            fkey: ForeignKey = ForeignKey('')
            for fkey in s.foreign_keys:
                # s)ource.column
                # t)arget.column
                c = fkey.column
                t = fkey.parent.table
                f = fkey.parent

                print("...", c, "->", f)

                waiting.appendleft(t)
                pass
        print("done")
    # end

    def scan(self, table: Table):

        # print("... primary_key")
        # print("... ...", table.primary_key)
        # print("... foreign_keys")
        # for fkey in table.foreign_keys:
        #     print("... ...", fkey)
        # print("... constraints")
        # for con in table.constraints:
        #     print("... ...", con)
        #
        # print("... foreign_key_constraints")
        # for fkc in table.foreign_key_constraints:
        #     print("... ...", fkc)
        # print("done")

        processed = set()
        waiting = deque([table])

        while len(waiting) > 0:
            s = waiting.pop()
            if s in processed:
                continue
            processed.add(s)

            fkey: ForeignKey = ForeignKey('')
            for fkey in s.foreign_keys:
                # s)ource.column
                # t)arget.column
                c = fkey.column
                t = fkey.parent.table
                f = fkey.parent

                print("...", c, "->", f)

                waiting.appendleft(t)
                pass

        print(table)
    # end

    # -----------------------------------------------------------------------
    # Metadata

    def _load_metadata(self):
        self.log.info("... loading metadata")
        self.metadata = MetaData()
        self.metadata.reflect(self.engine)
        # print(self.metadata.tables.keys())
        # print("n tables:", len(self.metadata.tables.keys()))

        # -------------------------------------------------------------------
        # Main data
        # -------------------------------------------------------------------

        # Data Model
        # self.iDataModelMaster \
        #     = Table('tb_idata_model_master', self.metadata, autoload_with=self.engine)
        self.iDataModelMaster = self.metadata.tables["tb_idata_model_master"]
        # [tb_idata_model_master]
        # id (bigint)
        # description (varchar(256))

        # self.iDataModelDetail \
        #     = Table('tb_idata_model_detail', self.metadata, autoload_with=self.engine)
        self.iDataModelDetail = self.metadata.tables["tb_idata_model_detail"]
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
        # self.iDataModuleMaster \
        #     = Table('tb_idata_module_master', self.metadata, autoload_with=self.engine)
        self.iDataModuleMaster = self.metadata.tables["tb_idata_module_master"]
        # [tb_idata_module_master]
        # id (bigint)
        # module_id (varchar(20))
        # module_description (varchar(200))

        # self.iDataModuleDetail \
        #     = Table('tb_idata_module_details', self.metadata, autoload_with=self.engine)
        self.iDataModuleDetail = self.metadata.tables["tb_idata_module_details"]
        # [tb_idata_module_details]
        # id (bigint)
        # master_id (bigint)
        # idata_id (bigint)
        # display_name (varchar(100))
        # is_enabled (varchar(1))

        # iData Master
        # self.iDataMaster \
        #     = Table('tb_idata_master', self.metadata, autoload_with=self.engine)
        self.iDataMaster = self.metadata.tables["tb_idata_master"]
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

        # self.iDataValuesMaster \
        #     = Table('tb_idata_values_master', self.metadata, autoload_with=self.engine)
        self.iDataValuesMaster = self.metadata.tables["tb_idata_values_master"]
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

        # self.iDataValuesDetail \
        #     = Table('tb_idata_values_detail', self.metadata, autoload_with=self.engine)
        self.iDataValuesDetail = self.metadata.tables["tb_idata_values_detail"]
        # [tb_idata_values_detail]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)

        # self.iDataValuesDetailHistory \
        #     = Table('tb_idata_values_detail_hist', self.metadata, autoload_with=self.engine)
        self.iDataValuesDetailHistory = self.metadata.tables["tb_idata_values_detail_hist"]
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
        # self.AttributeMaster \
        #     = Table('tb_attribute_master', self.metadata, autoload_with=self.engine)
        self.AttributeMaster = self.metadata.tables["tb_attribute_master"]
        # [tb_attribute_master]
        # id (bigint)
        # attribute_master_name (varchar(256))
        # attribute_desc (varchar(256))
        # createdby (varchar(256))
        # createddate (date)
        # hierarchy_type (bigint)

        # self.AttributeDetail \
        #     = Table('tb_attribute_detail', self.metadata, autoload_with=self.engine)
        self.AttributeDetail = self.metadata.tables["tb_attribute_detail"]
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
        # self.iPredictMasterFocussed \
        #     = Table('tb_ipr_conf_master_focussed', self.metadata, autoload_with=self.engine)
        self.iPredictMasterFocussed = self.metadata.tables["tb_ipr_conf_master_focussed"]
        # [tb_ipr_conf_master_focussed]
        # id (bigint)
        # ipr_conf_master_name (varchar(256))
        # ipr_conf_master_desc (varchar(256))
        # idata_model_details_id_fk (bigint)
        # area_id_fk (bigint)
        # skill_id_fk (bigint)
        # idata_id_fk (bigint)

        # self.iPredictDetailFocussed \
        #     = Table('tb_ipr_conf_detail_focussed', self.metadata, autoload_with=self.engine)
        self.iPredictDetailFocussed = self.metadata.tables["tb_ipr_conf_detail_focussed"]
        # [tb_ipr_conf_detail_focussed]
        # id (bigint)
        # parameter_desc (varchar(256))
        # parameter_value (varchar(256))
        # ipr_conf_master_id (bigint)
        # parameter_id (bigint)
        # to_populate (bigint)
        # skill_id_fk (bigint)
        # period (varchar(256))

        # self.iPredictModelDetailFocussed \
        #     = Table('tb_ipr_model_detail_focussed', self.metadata, autoload_with=self.engine)
        self.iPredictModelDetailFocussed = self.metadata.tables["tb_ipr_model_detail_focussed"]
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

        # self.iPredictTrainDataFocussed \
        #     = Table('tb_ipr_train_data_focussed', self.metadata, autoload_with=self.engine)
        self.iPredictTrainDataFocussed = self.metadata.tables["tb_ipr_train_data_focussed"]
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

        # self.iDataValuesMaster \
        #     = Table('tb_idata_values_master', self.metadata, autoload_with=self.engine)
        self.iDataValuesMaster = self.metadata.tables["tb_idata_values_master"]
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

        # self.iDataValuesDetail \
        #     = Table('tb_idata_values_detail', self.metadata, autoload_with=self.engine)
        self.iDataValuesDetail = self.metadata.tables["tb_idata_values_detail"]
        # [tb_idata_values_detail]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)

        # self.iDataValuesDetailHist \
        #     = Table('tb_idata_values_detail_hist', self.metadata, autoload_with=self.engine)
        self.iDataValuesDetailHist = self.metadata.tables["tb_idata_values_detail_hist"]
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