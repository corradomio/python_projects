from sqlalchemy import update

from .datamodels import *


# ---------------------------------------------------------------------------
# IDataMaster   (DataMaster)
# ---------------------------------------------------------------------------

PERIOD_LIT_TO_NAME_MAP = {
    'D': 'day',
    'W': 'week',
    'M': 'month'
}

PERIOD_NAME_TO_LIT_MAP = {
    'day': 'D',
    'week': 'W',
    'month': 'M'
}


class DataMaster(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.iDataMaster)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name

        self.check_data()
        return self.data['description']

    @property
    def description(self) -> str:
        self.check_data()
        return self.data['description']

    @property
    def data_model(self) -> DataModel:
        self.check_data()
        data_model_id = self.data['idatamodel_id_fk']
        return self.ipom.data_models().data_model(data_model_id)

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        self.check_data()
        area_hierarchy_id = self.data['area_id_fk']
        return self.ipom.hierachies().area_hierarchy(area_hierarchy_id)

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        self.check_data()
        skill_hierarchy_id = self.data['skill_id_fk']
        return self.ipom.hierachies().skill_hierarchy(skill_hierarchy_id)

    @property
    def period_hierarchy(self) -> PeriodHierarchy:
        self.check_data()
        period_hierarchy = self.data['period_hierarchy']
        period_length = self.data['period']
        return PeriodHierarchy(self.ipom, period_hierarchy, period_length)

    # -----------------------------------------------------------------------

    def delete(self):
        if self._id == NO_ID:
            return self

        self._name = self.name
        self._delete_data_master(self._id)
        super().delete()
        return self
    # end

    def _delete_data_master(self, data_master_id: int):

        # data_master
        #   TS Focussed
        #       TSF Feature
        #   Plan
        #       Train Data
        #       Predict Data

        with self.engine.connect() as conn:
            # 1) delete dependencies
            table = self.ipom.iDataValuesMaster
            query = select(table.c.id).where(table.c['idata_master_fk'] == data_master_id)
            self.log.debug(query)
            rlist = conn.execute(query)#.fetchall()
            plan_ids = [res[0] for res in rlist]

            # delete historical data
            table = self.ipom.iDataValuesDetailHist
            query = delete(table).where(table.c['value_master_fk'].in_(plan_ids))
            self.log.debug(query)
            conn.execute(query)

            # delete prediction data
            table = self.ipom.iDataValuesDetail
            query = delete(table).where(table.c['value_master_fk'].in_(plan_ids))
            self.log.debug(query)
            conn.execute(query)

            # delete plans
            table = self.ipom.iDataValuesMaster
            query = delete(table).where(table.c.id.in_(plan_ids))
            self.log.debug(query)
            conn.execute(query)

            # delete TS focussed
            table = self.ipom.iPredictMasterFocussed
            update(table).where(table.c['idata_id_fk'] == data_master_id).values(idata_id_fk=None)
            self.log.debug(query)
            conn.execute(query)

            # 2) delete data master
            table = self.ipom.iDataMaster
            query = delete(table).where(table.c.id == data_master_id)
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
    # end

    def create(self,
               data_model: Union[int, str],
               area_hierarchy: Union[int, str],
               skill_hierarchy: Union[int, str],
               freq: Literal['D', 'W', 'M'] = 'D',
               periods: int = 90):

        assert is_instance(self._name, str), "Missing Data Master name"

        assert is_instance(data_model, Union[int, str])
        assert is_instance(area_hierarchy, Union[int, str])
        assert is_instance(skill_hierarchy, Union[int, str])
        assert is_instance(freq, Literal['D', 'W', 'M'])
        assert is_instance(periods, int) and periods > 0

        if self._id != NO_ID:
            self.log.warning(f"Data Master '{self._name}' already existent")
            return self

        self._id = self._create_data_master(
            self._name,
            data_model=data_model,
            area_hierarchy=area_hierarchy,
            skill_hierarchy=skill_hierarchy,
            freq=freq,
            periods=periods
        )

        self._name = None
        return self
    # end

    def _create_data_master(
            self,
            name: str,
            data_model: Union[int, str],
            area_hierarchy: Union[int, str],
            skill_hierarchy: Union[int, str],
            freq: Literal['D', 'W', 'M'],
            periods: int = 90):
        """
        Create a Data Master

        :param name: name of the Data Master
        :param data_model: Data Model to use
        :param area_hierarchy: Area Hierarchy to use
        :param skill_hierarchy: Skill Hirerachy to use
        :param freq: Period Hierarchy to use
        :param periods: period length to use
        :return:
        """
        assert is_instance(freq, Literal['D', 'W', 'M'])
        assert is_instance(periods, int) and periods > 0

        data_model_id = self.ipom.data_models().data_model(data_model).id
        area_hierarchy_id = self.ipom.hierachies().area_hierarchy(area_hierarchy).id
        skill_hierarchy_id = self.ipom.hierachies().skill_hierarchy(skill_hierarchy).id
        period_hierarchy = PERIOD_LIT_TO_NAME_MAP[freq]

        # data_master_id = self.ipom._convert_id(name, table, ['description'], nullable=True)

        with self.engine.connect() as conn:
            table = self.ipom.iDataMaster
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
        return data_master_id
    # end
# end


class DataMasters(IPlanObject):

    def __init__(self, ipom):
        super().__init__(ipom)

    def data_master(self, id: Union[int, str]) -> DataMaster:
        data_master_id = self.ipom._convert_id(id, self.ipom.iDataMaster, ['description'], nullable=True)
        if data_master_id is None:
            return DataMaster(self.ipom, id)
        else:
            return DataMaster(self.ipom, data_master_id)

    def find_data_master(self,
                         data_model: Union[int, str],
                         area_hierarchy: Union[int, str],
                         skill_hierarchy: Union[int, str]) -> Optional[DataMaster]:
        """
        Find a Data Master having the selected Data Model, Area Hierarchy, Skill Hierarchy (and Period Hierarchy)
        Note: if there are multiple Data Masters, it is selected the first one.

        :param data_model: Data Model
        :param area_hierarchy: Area Hierarchy
        :param skill_hierarchy: Skill Hierarchy
        :return: Data Model or None
        """

        data_model_id = self.ipom._convert_id(data_model, self.ipom.iDataModelMaster, ['description'])
        area_hierarchy_id = self.ipom._convert_id(area_hierarchy, self.ipom.AttributeMaster, ['attribute_master_name', 'attribute_desc'])
        skill_hierarchy_id = self.ipom._convert_id(skill_hierarchy, self.ipom.AttributeMaster, ['attribute_master_name', 'attribute_desc'])

        with self.engine.connect() as conn:
            table = self.ipom.iDataMaster
            query = select(table.c.id).distinct().where((table.c['area_id_fk'] == area_hierarchy_id) &
                                                        (table.c['skill_id_fk'] == skill_hierarchy_id) &
                                                        (table.c['idatamodel_id_fk'] == data_model_id))
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            if len(rlist) == 1:
                return DataMaster(self.ipom, to_data(rlist[0]))
            elif len(rlist) == 0:
                self.log.error(f"No Data Master found with ({data_model_id},{area_hierarchy_id},{skill_hierarchy_id})")
                return None
            else:
                self.log.error(
                    f"Multiple Data Masters with found with (dara_model:{data_model_id},area_hierarchy:{area_hierarchy_id},skill_hierarchy:{skill_hierarchy_id})")
                return DataMaster(self.ipom, to_data(rlist[-1]))
# end
