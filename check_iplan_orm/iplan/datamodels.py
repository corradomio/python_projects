from stdlib import as_list
from .hierarchies import *


# ---------------------------------------------------------------------------
# IDataModelMaster  (DataModel)
# IDataModelDetail  (Measure)
# ---------------------------------------------------------------------------

class Measure(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.iDataModelDetail)
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


class DataModel(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.iDataModelMaster)

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
    def measures(self) -> list[Measure]:
        return self.details()

    def measure_ids(self, with_name=False) -> Union[list[int], dict[int, str]]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataModelDetail
            query = select(table.c.id, table.c['measure_id']).where(table.c['data_model_id_fk'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            if with_name:
                return {res[0]: res[1] for res in rlist}
            else:
                return [res[0] for res in rlist]
    # end

    def details(self) -> list[Measure]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataModelDetail
            query = select(table).where(table.c['data_model_id_fk'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        return [Measure(self.ipom, to_data(result)) for result in rlist]

    def measure(self, id: Union[int, str]) -> Measure:
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
            return Measure(self.ipom, to_data(ret))
    # end

    # -----------------------------------------------------------------------

    def delete(self):
        if self._id == NO_ID:
            return self

        self._name = self.name

        self._delete_data_model(self._id)

        super().delete()
        return self

    def _delete_data_model(self, data_model_id: int):
        with self.engine.connect() as conn:
            # 0) delete dependencies
            # query = text("""
            # DELETE FROM tb_ipr_conf_detail_focussed AS ticdf
            # WHERE ticdf.parameter_id IN (
            #     SELECT timd.id FROM tb_idata_model_detail AS timd
            #     WHERE timd.data_model_id_fk = :data_model_id
            # )
            # """)
            # self.log.debug(query)
            # conn.execute(query, parameters=dict(
            #     data_model_id=data_model_id
            # ))
            #
            # query = text("""
            # DELETE FROM tb_ipr_conf_master_focussed AS ticmf
            # WHERE ticmf.idata_model_details_id_fk = :data_model_id
            # """)
            # table = self.ipom.iPredictMasterFocussed
            # query = delete(table).where(
            #     table.c['idata_model_details_id_fk'] == data_model_id
            # )
            # self.log.debug(query)
            # conn.execute(query, parameters=dict(
            #     data_model_id=data_model_id
            # ))
            #
            # query = text("""
            # DELETE FROM tb_idata_values_master AS tivm
            # WHERE tivm.idata_master_fk IN (
            #     SELECT tim.id FROM tb_idata_master AS tim
            #     WHERE tim.idatamodel_id_fk = :data_model_id
            # )
            # """)
            # self.log.debug(query)
            # conn.execute(query, parameters=dict(
            #     data_model_id=data_model_id
            # ))
            #
            # query = text("""
            # DELETE FROM tb_idata_master AS tim
            # WHERE tim.idatamodel_id_fk = :data_model_id
            # """)
            # self.log.debug(query)
            # conn.execute(query, parameters=dict(
            #     data_model_id=data_model_id
            # ))

            # 1) delete tb_data_model_detail
            table = self.ipom.iDataModelDetail
            query = delete(table).where(
                table.c['data_model_id_fk'] == data_model_id
            )
            self.log.debug(query)
            conn.execute(query)

            # 2) delete tb_data_model_master
            table = self.ipom.iDataModelMaster
            query = delete(table).where(
                table.c.id == data_model_id
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return
    # end

    def create(self,
               targets: Union[str, list[str]],
               inputs: Union[None, str, list[str]]):
        if self._id != NO_ID:
            self.log.warning(f"Data Model '{self._name}' already existent")
            return self

        assert is_instance(self._name, str), "Missing Data Model name"

        self._id = self._create_data_model(
            targets=targets,
            inputs=inputs)
        self._name = None
        return self

    def _create_data_model(
        self, *,
        targets: Union[str, list[str]],
        inputs: Union[None, str, list[str]]):
        """

        :param name: Data Model name
        :param targets: measures used as FEED
        :param inputs: measures used as INPUT
        :return:
        """
        assert is_instance(targets, Union[str, list[str]])
        assert is_instance(inputs, Union[None, str, list[str]])

        name = self._name
        targets = as_list(targets, 'targets')
        inputs = as_list(inputs, 'inputs')

        # ensure that inputs DOESN'T contain targets
        common = set(inputs).intersection(targets)
        if len(common) > 0:
            self.log.warning(f"'inputs' columns contain some 'target' columns: {common}.Removed from 'inputs'")
            inputs = list(set(inputs).difference(targets))

        now = datetime.now()
        ntargets = len(targets)

        with self.engine.connect() as conn:
            # 1) create data model master
            table = self.ipom.iDataModelMaster
            query = insert(table).values(
                description=name,
            ).returning(table.c.id)
            self.log.debug(query)
            data_model_id: int = conn.execute(query).scalar()

            # 2) create data model detail
            table = self.ipom.iDataModelDetail
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
        return data_model_id
    # end
# end


class DataModels(IPlanObject):
    def __init__(self, ipom):
        super().__init__(ipom)

    def data_model(self, id: Union[int, str]) -> DataModel:
        data_model_id = self._convert_id(id, self.ipom.iDataModelMaster, ['description'], nullable=True)
        if data_model_id is None:
            return DataModel(self.ipom, id)
        else:
            return DataModel(self.ipom, data_model_id)
# end
