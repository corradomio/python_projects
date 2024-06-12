from sqlalchemy import MetaData, Engine, create_engine, URL

from .tseries import *


# ---------------------------------------------------------------------------
# IPlanObjectModel
# ---------------------------------------------------------------------------

class IPlanObjectModel(IPlanObject):

    @staticmethod
    def connect_using(url: Union[str, dict, URL], **kwargs):
        ipom = IPlanObjectModel(url, **kwargs)
        return ipom.connect()

    def __init__(self, url: Union[str, dict, URL], **kwargs):
        assert is_instance(url, Union[str, dict, URL])

        if is_instance(url, dict):
            url_props: dict = {} | url
            kwargs = url_props | kwargs

            if 'readonly' in url_props:
                del url_props['readonly']

            url = URL.create(**url_props)

        # NOTE: keep this orser
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
    # Properties

    @property
    def read_only(self):
        return self.kwargs.get('read_only', False)

    # -----------------------------------------------------------------------
    # Area/Skill hierarchy

    def hierachies(self) -> AttributeHierarchies:
        return AttributeHierarchies(self)

    # -----------------------------------------------------------------------
    # Data Model

    def data_models(self) -> DataModels:
        return DataModels(self)

    # -----------------------------------------------------------------------
    # Data Master

    def data_masters(self) -> DataMasters:
        return DataMasters(self)

    # -----------------------------------------------------------------------
    # Prediction Plan

    def plans(self) -> PredictionPlans:
        return PredictionPlans(self)

    # -----------------------------------------------------------------------
    # Time Series Focussed

    def time_series(self):
        return IPlanTimeSeries(self)

    # -----------------------------------------------------------------------

    def _load_metadata(self):
        self.metadata = MetaData()
        self.metadata.reflect(self.engine)

        # -------------------------------------------------------------------
        # Main data
        # -------------------------------------------------------------------

        # Data Model
        self.iDataModelMaster = self.metadata.tables["tb_idata_model_master"]
        # [tb_idata_model_master]
        # id (bigint)
        # description (varchar(256))

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
        self.iDataModuleMaster = self.metadata.tables["tb_idata_module_master"]
        # [tb_idata_module_master]
        # id (bigint)
        # module_id (varchar(20))
        # module_description (varchar(200))

        self.iDataModuleDetail = self.metadata.tables["tb_idata_module_details"]
        # [tb_idata_module_details]
        # id (bigint)
        # master_id (bigint)
        # idata_id (bigint)
        # display_name (varchar(100))
        # is_enabled (varchar(1))

        # iData Master
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

        # Area/Skill Hierarchies
        self.AttributeMaster = self.metadata.tables["tb_attribute_master"]
        # [tb_attribute_master]
        # id (bigint)
        # attribute_master_name (varchar(256))
        # attribute_desc (varchar(256))
        # createdby (varchar(256))
        # createddate (date)
        # hierarchy_type (bigint)

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

        self.iDataValuesDetail = self.metadata.tables["tb_idata_values_detail"]
        # [tb_idata_values_detail]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)

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

        self.iPredictModelDetailFocussed = self.metadata.tables["tb_ipr_model_detail_focussed"]
        # [tb_ipr_model_detail_focussed]
        # id(bigint)
        # best_model(text)
        # best_model_name(text)
        # best_r_2(double precision)
        # best_wape(double precision)
        # ohmodels_catftr(text)
        # area_id_fk(bigint)
        # ipr_conf_master_id_fk(bigint)
        # skill_id_fk(bigint)

        self.iPredictTestPredictionValuesFocussed = self.metadata.tables["tb_ipr_test_prediction_values_detail_focussed"]
        # ipr_conf_master_id_fk (int8)
        # area_id_fk (int8)
        # skill_id_fk (int8)
        # model_detail_id_fk (int8)
        # actual (numeric)
        # predicted (numeric)
        # state_date (date)

        self.iPredictPredictedValuesFocussed = self.metadata.tables["tb_ipr_predicted_values_detail_focussed"]
        # ipr_conf_master_id_fk (int8)
        # area_id_fk (int8)
        # skill_id_fk (int8)
        # model_detail_id_fk (int8)
        # actual (numeric)
        # predicted (numeric)
        # state_date (date)

        #
        # OLD/UNUSED TABLES
        #
        # self.iPredictTrainDataFocussed = self.metadata.tables["tb_ipr_train_data_focussed"]
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

        self.iDataValuesDetail = self.metadata.tables["tb_idata_values_detail"]
        # [tb_idata_values_detail]
        # id (bigint)
        # value_master_fk (bigint)
        # state_date (date)
        # updated_date (date)
        # model_detail_id_fk (bigint)
        # skill_id_fk (bigint)
        # value (double precision)

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
                rlist = conn.execute(query).fetchall()
                if len(rlist) > 0:
                    # [(id,)]
                    return rlist[0][0]
                continue
        if nullable:
            return None
        raise ValueError(f"Unable to convert '{what}' into an id using {table.name}")

    def _convert_name(self, id: int, table: Table, namecol: str, idcol: str = 'id') -> str:
        """

        :param id: id to convert
        :param table: table to use
        :param namecol: column containing the name
        :param idcol: column containing the 'id' value
        :return: the found name
        """
        assert is_instance(id, int)

        with self.engine.connect() as conn:
            query = select(table.c[namecol]).where(table.c[idcol] == id)
            result = conn.execute(query).scalar()
            return result
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
