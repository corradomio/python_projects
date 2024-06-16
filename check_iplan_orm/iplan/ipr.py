from .plans import *


# ---------------------------------------------------------------------------
# IPredictDetailFocussed
# ---------------------------------------------------------------------------

class IPredictDetailFocussed(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.iPredictDetailFocussed)

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
        return Measure(self.ipom, measure_id)
# end


class IPredictMasterFocussed(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.iPredictMasterFocussed)
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

    @property
    def data_model(self) -> DataModel:
        if self._data_model is None:
            data_model_id = self.data['idata_model_details_id_fk']
            self._data_model = self.ipom.data_models().data_model(data_model_id)
        return self._data_model

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        if self._area_hierarchy is None:
            area_hierarchy_id = self.data['area_id_fk']
            self._area_hierarchy = self.ipom.hierachies().area_hierarchy(area_hierarchy_id)
        return self._area_hierarchy

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        if self._skill_hierarchy is None:
            skill_hierarchy_id = self.data['skill_id_fk']
            self._skill_hierarchy = self.ipom.hierachies().skill_hierarchy(skill_hierarchy_id)
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
            rlist = conn.execute(query)#.fetchall()
            for res in rlist:
                id = res[0]
                type = res[1]
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
            return self._select_measure_names(measure_ids)
        else:
            return measure_ids

    def _select_measure_names(self, measure_ids: list[int]) -> dict[int, str]:
        mdict = {}
        with self.engine.connect() as conn:
            tmeasure = self.ipom.iDataModelDetail
            query = select(tmeasure.c['id', 'measure_id']).where(tmeasure.c['id'].in_(measure_ids))
            self.logsql.debug(query)
            rlist = conn.execute(query)
            for id, name in rlist:
                mdict[id] = name
        return mdict

    # Note: the COLUMN 'tb_ipr_conf_master_focussed.idata_id_fk' IS NULL
    #       in ALL records in the table!
    #       HOWEVER, in "theory" it is possible to retrieve teh 'Data Master' based on the
    #       values:  (data_model_id, area_hierarchy_id, skill_hierarchy_id)

    def details(self) -> list[IPredictDetailFocussed]:
        with self.engine.connect() as conn:
            tdetail = self.ipom.iPredictDetailFocussed
            query = select(tdetail).where(tdetail.c['ipr_conf_master_id'] == self.id)
            self.log.debug(f"{query}")
            rlist = conn.execute(query)#.fetchall()
            # idlist: [(id,), ...]
            return [IPredictDetailFocussed(self.ipom, to_data(result)) for result in rlist]

    # -----------------------------------------------------------------------

    def delete(self):
        if self._id == NO_ID:
            return self
        self._name = self.name
        self._delete_time_series_focussed(self._id)
        self._id = NO_ID
        return self

    def _delete_time_series_focussed(self, id: Union[int, str]):
        tsf_id = self.ipom._convert_id(id, self.ipom.iPredictMasterFocussed,
                                       ['ipr_conf_master_name', 'ipr_conf_master_desc'],
                                       nullable=True)
        if tsf_id is None:
            return

        with self.engine.connect() as conn:
            # 1) delete tb_ipr_conf_detail_focussed
            table = self.ipom.iPredictDetailFocussed
            query = delete(table).where(table.c['ipr_conf_master_id'] == tsf_id)
            self.logsql.debug(query)
            conn.execute(query)

            # 2) delete tb_ipr_conf_master_focussed
            table = self.ipom.iPredictMasterFocussed
            query = delete(table).where(table.c['id'] == tsf_id)
            self.logsql.debug(query)
            conn.execute(query)
            conn.commit()
        return
    # end

    def create(self,
               targets: Union[str, list[str]],
               inputs: Union[None, str, list[str]] = None,
               data_master: Union[None, int, str] = None,
               description: Optional[str] = None
               ):

        assert is_instance(targets, Union[str, list[str]])
        assert is_instance(inputs, Union[None, str, list[str]])
        assert is_instance(data_master, Union[None, int, str])
        assert is_instance(description, Optional[str])

        assert data_master is not None or self._data_master is not None

        targets = as_list(targets, "targets")
        inputs = as_list(inputs, "inputs")

        if data_master is not None:
            self._data_master = self.ipom.data_masters().data_master(data_master)

        assert isinstance(self._name, str)
        self._id = self._create_time_series_focussed(
            self._name,
            targets=targets,
            inputs=inputs,
            # data_master=data_master,
            description=description,
        )
        self._name = None

    def _create_time_series_focussed(self, name: str,
                                     # data_master: Union[int, str], *,
                                     targets: list[str],
                                     inputs: list[str],
                                     description: Optional[str] = None) -> int:
        """
        Create a time series

        :param name: Time Series name
        :param targets: list of target measures
        :param inputs: list of input measures
        :param description:  Time Series description
        """

        assert is_instance(targets, list[str] )
        assert is_instance(inputs, list[str] )

        data_master = self._data_master
        data_model: DataModel = data_master.data_model
        data_model_id = data_model.id
        area_hierarchy_id = data_master.area_hierarchy.id
        skill_hierarchy_id = data_master.skill_hierarchy.id

        targets = as_list(targets, 'targets')
        inputs = as_list(inputs, 'inputs')
        description = name if description is None else description

        # create the tb_ipr_conf_master_focussed
        with self.engine.connect() as conn:
            # 1) fill tb_ipr_conf_master_focussed
            table = self.ipom.iPredictMasterFocussed
            query = insert(table).values(
                ipr_conf_master_name=name,
                ipr_conf_master_desc=description,
                idata_model_details_id_fk=data_model_id,
                area_id_fk=area_hierarchy_id,
                skill_id_fk=skill_hierarchy_id,
                idata_id_fk=None
            ).returning(table.c.id)
            self.logsql.debug(query)
            tsf_id = conn.execute(query).scalar()
            # 2) fill tb_ipr_conf_detail_focussed
            table = self.ipom.iPredictDetailFocussed
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
        return tsf_id
# end
