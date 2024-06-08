from typing import Union, Literal
from sqlalchemy import *
from stdlib import is_instance


def delete_area_hierarchy(self, id: Union[int, str]):
    self.delete_attribute_hierarchy(id, 'area')

def delete_skill_hierarchy(self, id: Union[int, str]):
    self.delete_attribute_hierarchy(id, 'skill')

def delete_attribute_hierarchy(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']):
    assert is_instance(hierarchy_type, Literal['area', 'skill'])
    area_hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'],
                                         nullable=True)
    if area_hierarchy_id is None:
        return

    htype = self.hierarchy_type(area_hierarchy_id)
    assert htype == hierarchy_type, f"Invalid hierarchy {id}: required '{hierarchy_type}', found '{htype}'"

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

--

def create_area_hierarchy(self, name: str, hierarchy_tree) \
        -> AttributeHierarchy:
    return self.create_attribute_hierarchy(name, hierarchy_tree, 'area')

def create_skill_hierarchy(self, name: str, hierarchy_tree) \
        -> AttributeHierarchy:
    return self.create_attribute_hierarchy(name, hierarchy_tree, 'skill')

def create_attribute_hierarchy(self, name: str, hierarchy_tree, hierarchy_type: Literal['area', 'skill']) \
        -> int:
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
        -> int:
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
    # return AttributeHierarchy(self, hierarchy_id, self.AttributeMaster)
    return hierarchy_id
# end

-----------------------------------------------------------------------

def hierarchy_type(self, id: Union[int, str]) -> Literal["area", "skill"]:
    hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'])

    with self.engine.connect() as conn:
        table = self.AttributeMaster
        query = select(table.c['hierarchy_type']).where(table.c['id'] == hierarchy_id)
        self.log.debug(f"{query}")
        hierarchy_type = conn.execute(query).fetchone()[0]
    return "area" if hierarchy_type == 1 else "skill"

@deprecated
def area_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
    return self.attribute_hierarchy(id, "area")

@deprecated
def area_feature(self, id: Union[int, str]) -> AttributeDetail:
    return self.attribute_detail(id, "area")

@deprecated
def skill_hierarchy(self, id: Union[int, str]) -> AttributeHierarchy:
    return self.attribute_hierarchy(id, "skill")

@deprecated
def skill_feature(self, id: Union[int, str]) -> AttributeDetail:
    return self.attribute_detail(id, "skill")

def attribute_hierarchy(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']) \
        -> AttributeHierarchy:
    hierarchy_id = self._convert_id(id, self.AttributeMaster, ['attribute_master_name', 'attribute_desc'],
                                    nullable=True)
    if hierarchy_id is None:
        hierarchy = AttributeHierarchy(self.ipom, NO_ID)
        hierarchy.set_name_type(str(id), hierarchy_type)
    else:
        hierarchy = AttributeHierarchy(self.ipom, hierarchy_id)
    assert hierarchy.type == hierarchy_type, f"Invalid hierarchy {id}: required '{hierarchy_type}', found '{hierarchy.type}'"
    return hierarchy

def attribute_detail(self, id: Union[int, str], hierarchy_type: Literal['area', 'skill']) \
        -> AttributeDetail:
    feature_id = self._convert_id(id, self.AttributeDetail, ['attribute', 'description'])
    detail = AttributeDetail(self.ipom, feature_id)
    assert self.hierarchy_type(detail.hierarchy_id) == hierarchy_type
    return detail


def data_model(self, id: Union[int, str]) -> IDataModel:
    data_model_id = self._convert_id(id, self.iDataModelMaster, ['description'])
    return IDataModel(self, data_model_id)

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
        return IDataModel(self, data_model_id)

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


def data_master(self, id: Union[int, str]) -> IDataMaster:
    data_master_id = self._convert_id(id, self.iDataMaster, ['description'])
    return IDataMaster(self, data_master_id)

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
            return IDataMaster(self.ipom, to_data(rlist[0]))
        elif len(rlist) == 0:
            self.log.error(f"No Data Master found with ({data_model_id},{area_hierarchy_id},{skill_hierarchy_id})")
            return None
        else:
            self.log.error(
                f"Multiple Data Masters with found with (dara_model:{data_model_id},area_hierarchy:{area_hierarchy_id},skill_hierarchy:{skill_hierarchy_id})")
            return IDataMaster(self.ipom, to_data(rlist[-1]))

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

    data_model_id = self.data_models().data_model(data_model).id
    area_hierarchy_id = self.hierachies().area_hierarchy(area_hierarchy).id
    skill_hierarchy_id = self.hierachies().skill_hierarchy(skill_hierarchy).id

    table = self.iDataMaster
    data_master_id = self._convert_id(name, table, ['description'], nullable=True)
    already_exists = data_master_id is not None
    if already_exists and not update:
        self.log.warning(f"Data Master {name} already existent")
        return IDataMaster(self, data_master_id)

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
    return data_master_id
# end

def delete_data_master(self, id: Union[int, str]):
    data_master_id = self._convert_id(id, self.iDataMaster, ['description'], nullable=True)
    if data_master_id is None:
        return

    # data_master
    #   TS Focussed
    #       TSF Feature
    #   Plan
    #       Train Data
    #       Predict Data

    with self.engine.connect() as conn:
        # 1) delete dependencies
        table = self.iDataValuesMaster
        query = select(table.c.id).where(table.c['idata_master_fk'] == data_master_id)
        self.log.debug(query)
        results = conn.execute(query).fetchall()
        plan_ids = [res[0] for res in results]

        # delete historical data
        table = self.iDataValuesDetailHist
        query = delete(table).where(table.c['value_master_fk'].in_(plan_ids))
        self.log.debug(query)
        conn.execute(query)

        # delete prediction data
        table = self.iDataValuesDetail
        query = delete(table).where(table.c['value_master_fk'].in_(plan_ids))
        self.log.debug(query)
        conn.execute(query)

        # delete plans
        table = self.iDataValuesMaster
        query = delete(table).where(table.c.id.in_(plan_ids))
        self.log.debug(query)
        conn.execute(query)

        # delete TS focussed
        table = self.iPredictMasterFocussed
        update(table).where(table.c['idata_id_fk'] == data_master_id).values(idata_id_fk=None)
        self.log.debug(query)
        conn.execute(query)

        # 2) delete data master
        table = self.iDataMaster
        query = delete(table).where(table.c.id == data_master_id)
        self.log.debug(query)
        conn.execute(query)
        conn.commit()
# end

def select_data_master_ids(self, data_model_id: int, area_hierarchy_id: int, skill_hierarchy_id: int) \
        -> list[int]:
    with self.engine.connect() as conn:
        table = self.ipom.iDataMaster
        query = select(table.c.id).where((table.c['area_id_fk'] == area_hierarchy_id) &
                                         (table.c['skill_id_fk'] == skill_hierarchy_id) &
                                         (table.c['idatamodel_id_fk'] == data_model_id))
        self.log.debug(f"{query}")
        rlist = conn.execute(query).fetchall()
        return [result[0] for result in rlist]


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
    data_master = self.data_masters().data_master(data_master_id)
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
        data_master_id: int = self.data_masters().data_master(data_master).id
    else:
        _, data_master_id = self._select_data_values_master_ids_by_plan(plan, area_feature_ids)

    return IPredictTimeSeries(self, pmf.id, data_master_id)

def predict_master_focussed(self, id: Union[int, str]) -> IPredictMasterFocussed:
    id = self._convert_id(id, self.iPredictMasterFocussed, ['ipr_conf_master_name', 'ipr_conf_master_desc'])
    return IPredictMasterFocussed(self, id)






# -----------------------------------------------------------------------

# def exists_plan(self, name_or_date: Union[str, datetime], data_master_id: int) -> bool:
#     assert is_instance(name_or_date, Union[str, datetime])
#
#     if isinstance(name_or_date, str):
#         name: str = name_or_date
#
#         with self.engine.connect() as conn:
#             table = self.ipom.iDataValuesMaster
#             query = select(func.count()).select_from(table).where(
#                 table.c.name.like(f"%{name}%") &
#                 (table.c['idata_master_fk'] == data_master_id)
#             )
#             self.log.debug(query)
#             count = conn.execute(query).scalar()
#     elif is_instance(name_or_date, datetime):
#         date: datetime = name_or_date
#
#         with self.engine.connect() as conn:
#             table = self.ipom.iDataValuesMaster
#             query = select(func.count()).select_from(table).where(
#                 (table.c['start_date'] <= date) &
#                 (table.c['end_date'] >= date) &
#                 (table.c['idata_master_fk'] == data_master_id)
#             )
#             self.log.debug(query)
#             count = conn.execute(query).scalar()
#     else:
#         raise ValueError(f"Unsupported type for {name_or_date}")
#
#     return count > 0

# def delete_plan(self, name: str):
#     assert is_instance(name, str)
#
#     with self.engine.connect() as conn:
#         table = self.ipom.iDataValuesMaster
#         query = delete(table).where(
#             table.c.name.like(f"%{name}%")
#         )
#         self.log.debug(query)
#         conn.execute(query)
#         conn.commit()

# def create_plan(self,
#                 name: Optional[str],
#                 data_master: Union[int, str, IDataMaster],
#                 start_date: Union[None, datetime, tuple[datetime, datetime]],
#                 end_date: Optional[datetime] = None,
#                 area_feature_ids: Union[None, int, list[int]] = None,
#                 force=False):
#     assert is_instance(name, Optional[str])
#     assert is_instance(data_master, Union[int, str, IDataMaster])
#     assert is_instance(start_date, Union[datetime, tuple[datetime, datetime]])
#     assert is_instance(end_date, Optional[datetime])
#     assert is_instance(area_feature_ids, Union[None, int, list[int]])
#
#     #
#     # prepare the data
#     #
#     now: datetime = datetime.now()
#     area_feature_ids = as_list(area_feature_ids)
#     """:type: list[int]"""
#
#     # retrieve the Data Master
#     if isinstance(data_master, int | str):
#         data_master = self.ipom.data_masters().data_master(data_master)
#     data_master_id = data_master.id
#
#     if name is None:
#         now_str = now.strftime('%Y-%m-%d %H:%M:%S')
#         name = f"Auto_Plan_OM_{now_str}"
#
#     already_exists = self.exists_plan(name, data_master_id)
#     if already_exists and not force:
#         self.log.warning(f"Plan {name} already existent")
#         return self
#     if already_exists:
#         self.log.warning(f"Delete plan {name}")
#         self.delete_plan(name)
#
#     af_ids: list[int] = data_master.area_hierarchy.feature_ids()
#     if len(area_feature_ids) == 0:
#         area_feature_ids = af_ids
#     else:
#         af_count = len(area_feature_ids)
#         ai_count = len(set(af_ids).intersection(area_feature_ids))
#         if af_count != ai_count:
#             self.log.error("Found incompatible area_feature_ids")
#             self.log.error(f"      data_master id: {data_master_id}")
#             self.log.error(f"   master's area ids: {list(af_ids)}")
#             self.log.error(f"    area_feature_ids: {area_feature_ids}")
#     # end
#
#     #
#     # parse (start_date, end_date)
#     #
#
#     # if end_date is not specified, it is computed as 'start_date' + period_length
#     if isinstance(start_date, tuple | list):
#         start_date, end_date = start_date
#
#     # compute end_date based on start_date & period_length
#     if end_date is None:
#         periods = data_master.period_hierarchy.periods
#         end_date = start_date + timedelta(days=periods)
#
#     #
#     # create the plans for each area
#     #
#
#     # [tb_idata_values_master]
#     # -- id
#     #  1) start_date
#     #  2) end_date
#     #  3) name
#     #  4) created_date
#     #  5) idata_master_fk
#     #  6) loan_updated_time
#     #  7) published
#     #  8) isscenario
#     #  9) temp_ind
#     # 10) area_id
#     # 11) last_updated_date
#     # 12) published_id
#     # 13) note
#
#     # STUPID implementation
#     count = 0
#     with (self.engine.connect() as conn):
#         table = self.ipom.iDataValuesMaster
#         for area_feature_id in area_feature_ids:
#             stmt = insert(table).values(
#                 start_date=start_date,
#                 end_date=end_date,
#                 name=name,
#                 created_date=now,
#                 idata_master_fk=data_master_id,
#                 loan_updated_time=now,
#                 published='N',
#                 isscenario='N',
#                 temp_ind='N',
#                 area_id=area_feature_id,
#                 last_updated_date=None,
#                 published_id=None,
#                 note="created by " + CREATED_BY
#             ).returning(table.c.id)
#             if count == 0: self.log.debug(stmt)
#             rec_id = conn.execute(stmt).scalar()
#             count += 1
#         conn.commit()
#     return
# # end




# class IDataValuesMaster(IPlanData):
#     def __init__(self, ipom, id):
#         super().__init__(ipom, id, ipom.iDataValuesMaster)
#         self.check_data()
#
#     @property
#     def name(self) -> str:
#         return self.data['name']
#
#     @property
#     def data_master(self) -> DataMaster:
#         data_master_id = self.data['idata_master_fk']
#         return self.ipom.data_masters().data_master(data_master_id)
#
#     # @property
#     # def data_model(self):
#     #     return self.data_master.data_model
#
#     @property
#     def area_hierarchy(self) -> AttributeHierarchy:
#         # check if the 'area_id' is consistent with the Area Hierarchy defined in Data Master
#         area_hierarchy_id = self.ipom.area_feature(self.data['area_id']).hierarchy_id
#         area_hierarchy = self.data_master.area_hierarchy
#         assert area_hierarchy_id == area_hierarchy.id
#         return area_hierarchy
#
#     @property
#     def skill_hierarchy(self) -> AttributeHierarchy:
#         return self.data_master.skill_hierarchy
#
#     @property
#     def start_date(self) -> datetime:
#         return self.data['start_date']
#
#     @property
#     def end_date(self) -> datetime:
#         return self.data['end_date']
#
#     @property
#     def area_feature(self) -> AttributeDetail:
#         area_feature_id = self.data['area_id']
#         return self.ipom.area_feature(area_feature_id)
#
#     def select_data_values(self):
#         return self.ipom.select_data_values(self.id)
# # end




    # def delete_train_data(self,
    #                       data_master_id: int,
    #                       plan_ids: list[int],
    #                       area_feature_dict: dict[int, str],
    #                       skill_feature_dict: dict[int, str],
    #                       measure_dict: dict[int, str]
    #                       ):
    #
    #     assert is_instance(data_master_id, int)
    #     assert is_instance(plan_ids, list[int])
    #     assert is_instance(area_feature_dict, dict[int, str])
    #     assert is_instance(skill_feature_dict, dict[int, str])
    #     assert is_instance(measure_dict, dict[int, str])
    #
    #     # 1) retrieve all area/skill feature ids
    #     area_feature_ids = list(area_feature_dict.keys())
    #     skill_feature_ids = list(skill_feature_dict)
    #     measure_ids = list(measure_dict.keys())
    #
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesDetailHist
    #
    #         # 2) retrieve the data with 'skill NOT NULL'
    #         query = delete(table) \
    #             .where(table.c['value_master_fk'].in_(plan_ids) &
    #                    table.c['model_detail_id_fk'].in_(measure_ids) &
    #                    table.c['area_id_fk'].in_(area_feature_ids) &
    #                    table.c['skill_id_fk'].in_(skill_feature_ids))
    #         self.log.debug(query)
    #         conn.execute(query)
    #         conn.commit()
    #     return
    # # end

    # def delete_predict_data(self,
    #                         data_master_id: int,
    #                         plan_ids: list[int],
    #                         area_feature_dict: dict[int, str],
    #                         skill_feature_dict: dict[int, str],
    #                         measure_dict: dict[int, str],
    #                         ):
    #
    #     assert is_instance(data_master_id, int)
    #     assert is_instance(plan_ids, list[int])
    #     assert is_instance(area_feature_dict, dict[int, str])
    #     assert is_instance(skill_feature_dict, dict[int, str])
    #     assert is_instance(measure_dict, dict[int, str])
    #     # assert is_instance(start_date, Optional[datetime])
    #     # assert is_instance(end_date, Optional[datetime])
    #     # assert is_instance(freq, Literal['D', 'W', 'M'])
    #
    #     # 1) retrieve all area/skill feature ids
    #     area_feature_ids = list(area_feature_dict.keys())
    #     skill_feature_ids = list(skill_feature_dict)
    #     measure_ids = list(measure_dict.keys())
    #
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesDetail
    #         query = delete(table) \
    #             .where(table.c['value_master_fk'].in_(plan_ids) &
    #                    table.c['model_detail_id_fk'].in_(measure_ids) &
    #                    table.c['skill_id_fk'].in_(skill_feature_ids))
    #         self.log.debug(query)
    #         conn.execute(query, parameters=dict(
    #             plan_ids=tuple(plan_ids),
    #             measure_ids=tuple(measure_ids),
    #             skill_feature_ids=tuple(skill_feature_ids),
    #             area_feature_ids=tuple(area_feature_ids)
    #         ))
    #         conn.commit()
    #     return
    # # end



    # -----------------------------------------------------------------------

    # def select_data_values(self, data_values_master_id) -> DataFrame:
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesDetail
    #         query = select(table.c['state_date', 'model_detail_id_fk', 'skill_id_fk', 'value']) \
    #             .where(table.c['value_master_fk'] == data_values_master_id)
    #         self.log.debug(query)
    #         df = pd.read_sql_query(query, self.engine)
    #     return df

    # def select_plan_ids(
    #     self,
    #     name: Optional[str],
    #     data_master_ids: list[int],
    #     area_feature_ids: list[int],
    # ) -> list[int]:
    #     # alias: select_plan_ids(...)
    #     table = self.iDataValuesMaster
    #     if name is None:
    #         query = select(table.c.id).where(
    #             table.c['idata_master_fk'].in_(data_master_ids) &
    #             table.c['area_id'].in_(area_feature_ids)
    #         )
    #         self.log.debug(query)
    #     else:
    #         query = select(table.c.id).where(
    #             (table.c['name'] == name) &
    #             table.c['idata_master_fk'].in_(data_master_ids) &
    #             table.c['area_id'].in_(area_feature_ids)
    #         )
    #         self.log.debug(query)
    #     with self.engine.connect() as conn:
    #         rlist = conn.execute(query)
    #         return [result[0] for result in rlist]

    # def select_data_values_master_date_interval(self, plan_ids: list[int]) \
    #         -> Optional[tuple[datetime, datetime]]:
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
    #             table.c.id.in_(plan_ids)
    #         )
    #         self.log.debug(query)
    #         result = conn.execute(query).fetchone()
    #         if result[0] is None or result[1] is None:
    #             return None
    #         else:
    #             return result[0], result[1]

    # def select_measure_names(self, measure_ids: list[int]) -> dict[int, str]:
    #     mdict = {}
    #     with self.engine.connect() as conn:
    #         tmeasure = self.iDataModelDetail
    #         query = select(tmeasure.c['id', 'measure_id']).where(tmeasure.c['id'].in_(measure_ids))
    #         self.log.debug(query)
    #         rlist = conn.execute(query)
    #         for id, name in rlist:
    #             mdict[id] = name
    #     return mdict




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

    # def select_train_data(
    #     self,
    #     data_master_id: int,
    #     plan_ids: list[int],  # data_values_master_ids
    #     area_feature_dict: dict[int, str],
    #     skill_feature_dict: dict[int, str],
    #     measure_dict: dict[int, str],
    #     new_format=False) -> DataFrame:
    #     """
    #     Retrieve the historical data from 'tb_idata_values_detail_hist' based on
    #
    #         - data_master_id
    #         - plan_ids
    #         - area_feature_ids
    #         - skill_feature_ids
    #         - measure_ids
    #
    #     It is possible to replace the area/skill/measure ids with the correspondent names
    #
    #     :param data_master_id:
    #     :param plan_ids:
    #     :param area_feature_dict:
    #     :param skill_feature_dict:
    #     :param measure_dict:
    #     :param new_format: if to create a dataframe compatible with
    #         the current implementation of the new format
    #     :return: a dataframe with the following columns
    #             if 'new_format == True':
    #                 columns: ['area:str', 'skill:str', 'date:datetime', <measure_1:float>, ...]
    #             else
    #                 columns: ['skill_id_fk:int', 'area_id_fk:int', 'time:datetime', 'day:str', <measure_1: float>, ...]
    #     """
    #
    #     assert is_instance(data_master_id, int)
    #     assert is_instance(plan_ids, list[int])
    #     assert is_instance(area_feature_dict, dict[int, str])
    #     assert is_instance(skill_feature_dict, dict[int, str])
    #     assert is_instance(measure_dict, dict[int, str])
    #
    #     # 1) retrieve all area/skill feature ids
    #     area_feature_ids = list(area_feature_dict.keys())
    #     skill_feature_ids = list(skill_feature_dict)
    #     measure_ids = list(measure_dict.keys())
    #
    #     table = self.iDataValuesDetailHist
    #
    #     # 2) retrieve the data with 'skill NOT NULL'
    #     query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
    #         .where(table.c['model_detail_id_fk'].in_(measure_ids) &
    #                table.c['area_id_fk'].in_(area_feature_ids) &
    #                table.c['skill_id_fk'].in_(skill_feature_ids))
    #     self.log.debug(query)
    #     df_with_skill = pd.read_sql_query(query, self.engine)
    #
    #     # 3) retrieve the data with 'skill IS NULL'
    #     query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
    #         .where(table.c['model_detail_id_fk'].in_(measure_ids) &
    #                table.c['area_id_fk'].in_(area_feature_ids) &
    #                (table.c['skill_id_fk'] == None)
    #                )
    #     self.log.debug(query)
    #     df_no_skill = pd.read_sql_query(query, self.engine)
    #
    #     # 4) concatenate df_with_skill WITH df_no_skill
    #     df = concatenate_no_skill_df(df_with_skill, df_no_skill, skill_feature_dict)
    #
    #     return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)
    # # end

    # def select_predict_data(
    #     self,
    #     data_master_id: int,
    #     plan_ids: list[int],  # data_values_master_ids
    #     area_feature_dict: dict[int, str],
    #     skill_feature_dict: dict[int, str],
    #     measure_dict: dict[int, str],
    #     new_format=False) -> DataFrame:
    #     """
    #
    #     :param data_master_id:
    #     :param plan_ids:
    #     :param area_feature_dict:
    #     :param skill_feature_dict:
    #     :param measure_dict:
    #     :param new_format:
    #     :return:
    #     """
    #
    #     assert is_instance(data_master_id, int)
    #     assert is_instance(plan_ids, list[int])
    #     assert is_instance(area_feature_dict, dict[int, str])
    #     assert is_instance(skill_feature_dict, dict[int, str])
    #     assert is_instance(measure_dict, dict[int, str])
    #
    #     # 1) retrieve all area/skill feature ids
    #     area_feature_ids = list(area_feature_dict.keys())
    #     skill_feature_ids = list(skill_feature_dict)
    #     measure_ids = list(measure_dict.keys())
    #
    #     # 2) retrieve the data with 'skill NOT NULL'
    #     qtext = """
    #             select tivm.area_id as area_id_fk,
    #                    tivd.skill_id_fk as skill_id_fk,
    #                    tivd.model_detail_id_fk as model_detail_id_fk,
    #                    tivd.state_date as state_date,
    #                    tivd.value as value
    #              from tb_idata_values_detail as tivd
    #              join tb_idata_values_master as tivm on tivm.id = tivd.value_master_fk
    #             where tivd.value_master_fk in :plan_ids
    #               and tivd.model_detail_id_fk in :measure_ids
    #               and tivd.skill_id_fk in :skill_feature_ids
    #               and tivm.area_id in :area_feature_ids
    #             """
    #     query = text(qtext)
    #     self.log.debug(query)
    #     df = pd.read_sql_query(query, self.engine, params=dict(
    #         plan_ids=tuple(plan_ids),
    #         measure_ids=tuple(measure_ids),
    #         skill_feature_ids=tuple(skill_feature_ids),
    #         area_feature_ids=tuple(area_feature_ids)
    #     ))
    #
    #     return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)

    # def select_predict_data_ext(
    #     self,
    #     data_master_id: int,
    #     plan_ids: list[int],  # data_values_master_ids
    #     area_feature_dict: dict[int, str],
    #     skill_feature_dict: dict[int, str],
    #     input_measure_ids: list[int],
    #     measure_dict: dict[int, str],
    #     start_date: Optional[datetime] = None,
    #     end_date: Optional[datetime] = None,
    #     freq: Literal['D', 'W', 'M'] = 'D',
    #     new_format=False) -> DataFrame:
    #     """
    #
    #     :param data_master_id:
    #     :param plan_ids:
    #     :param area_feature_dict:
    #     :param skill_feature_dict:
    #     :param input_measure_ids:
    #     :param measure_dict:
    #     :param start_date:
    #     :param end_date:
    #     :param freq:
    #     :param new_format:
    #     :return:
    #     """
    #
    #     assert is_instance(data_master_id, int)
    #     assert is_instance(plan_ids, list[int])
    #     assert is_instance(area_feature_dict, dict[int, str])
    #     assert is_instance(skill_feature_dict, dict[int, str])
    #     assert is_instance(input_measure_ids, list[int])
    #     assert is_instance(measure_dict, dict[int, str])
    #     assert is_instance(start_date, Optional[datetime])
    #     assert is_instance(end_date, Optional[datetime])
    #     assert is_instance(freq, Literal['D', 'W', 'M'])
    #
    #     # Note: the dataset contains all measures in 'input_measure_ids' or 'measure_dict'
    #     #   plus: 'area', 'skill', 'date'
    #
    #     # 1) retrieve all area/skill feature ids
    #     area_feature_ids = list(area_feature_dict.keys())
    #     skill_feature_ids = list(skill_feature_dict)
    #     measure_ids = list(measure_dict.keys())
    #
    #     # 2) retrieve start/end dates for each area
    #     start_end_date_dict = self._select_start_end_date_dict(plan_ids, area_feature_ids)
    #
    #     # add the DEFAULT start/end date for the areas without a date range
    #     # Note: it is used 0 (ZERO) as key
    #     if start_date is not None and end_date is not None:
    #         start_end_date_dict[0] = (start_date, end_date)
    #
    #     # Note: [tb_idata_values_detail]
    #     #   DOESNT' CONTAIN 'area_id_fk'
    #     #   BUT it has a reference with [tb_idata_values_master] ('value_master_fk)
    #     #   AND 'tb_idata_values_master' contains 'area_id', that is, the required 'area_id_fk'
    #     df = self._select_predict_data(plan_ids, area_feature_ids, skill_feature_ids, measure_ids)
    #
    #     df_pivoted = pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)
    #
    #     df_pivoted = compose_predict_df(
    #         df_pivoted,
    #         area_feature_dict,
    #         skill_feature_dict,
    #         input_measure_ids,
    #         measure_dict,
    #         start_end_date_dict,
    #         freq,
    #         new_format=new_format
    #     )
    #
    #     return df_pivoted
    # # end

    # -----------------------------------------------------------------------

    # def _select_start_end_date_dict(self, plan_ids, area_feature_ids) \
    #         -> dict[int, tuple[datetime, datetime]]:
    #     # data_values_master_ids -> plan_ids
    #     # 2) retrieve start/end dates for each area
    #
    #     # qtext = """
    #     #         select tivm.area_id as area_id_fk, tivm.start_date, tivm.end_date
    #     #           from tb_idata_values_master as tivm
    #     #          where tivm.id in :plan_ids
    #     #            and tivm.area_id in :area_feature_ids
    #     #         """
    #     # query = text(qtext)
    #
    #     table = self.iDataValuesMaster
    #     query = select(table.c['area_id', 'start_date', 'end_date']).where(
    #         table.c.id.in_(plan_ids) &
    #         table.c['area_id'].in_(area_feature_ids)
    #     )
    #     self.log.debug(query)
    #     midnight = time(0, 0, 0)
    #
    #     with self.engine.connect() as conn:
    #         rlist = conn.execute(query, parameters=dict(
    #             plan_ids=tuple(plan_ids),
    #             area_feature_ids=tuple(area_feature_ids)
    #         )).fetchall()
    #         start_end_date_dict = {
    #             r[0]: (
    #                 datetime.combine(r[1], midnight),
    #                 datetime.combine(r[2], midnight)
    #             )
    #             for r in rlist
    #         }
    #     return start_end_date_dict
    # # end

    # def _select_predict_data(self,
    #                          plan_ids: list[int],  # data_values_master_ids
    #                          area_feature_ids: list[int],
    #                          skill_feature_ids: list[int],
    #                          measure_ids: list[int]) \
    #         -> DataFrame:
    #     # qtext = """
    #     #         select tivm.area_id as area_id_fk,
    #     #                tivd.skill_id_fk as skill_id_fk,
    #     #                tivd.model_detail_id_fk as model_detail_id_fk,
    #     #                tivd.state_date as state_date,
    #     #                tivd.value as value
    #     #          from tb_idata_values_detail as tivd,
    #     #               tb_idata_values_master as tivm
    #     #         where tivd.value_master_fk in :plan_ids
    #     #           and tivd.model_detail_id_fk in :measure_ids
    #     #           and tivd.skill_id_fk in :skill_feature_ids
    #     #           and tivm.area_id in :area_feature_ids
    #     #           and tivm.id in :plan_ids
    #     #           and tivm.id = tivd.value_master_fk
    #     #         """
    #     qtext = """
    #             select tivm.area_id as area_id_fk,
    #                    tivd.skill_id_fk as skill_id_fk,
    #                    tivd.model_detail_id_fk as model_detail_id_fk,
    #                    tivd.state_date as state_date,
    #                    tivd.value as value
    #              from tb_idata_values_detail as tivd
    #              join tb_idata_values_master as tivm on tivm.id = tivd.value_master_fk
    #             where tivd.value_master_fk in :plan_ids
    #               and tivd.model_detail_id_fk in :measure_ids
    #               and tivd.skill_id_fk in :skill_feature_ids
    #               and tivm.area_id in :area_feature_ids
    #     """
    #     query = text(qtext)
    #     self.log.debug(query)
    #     df = pd.read_sql_query(query, self.engine, params=dict(
    #         plan_ids=tuple(plan_ids),
    #         measure_ids=tuple(measure_ids),
    #         skill_feature_ids=tuple(skill_feature_ids),
    #         area_feature_ids=tuple(area_feature_ids)
    #     ))
    #     return df
    # # end

    # def _select_data_values_master_ids(self, data_master_ids: list[int], area_feature_ids: list[int]) \
    #         -> list[int]:
    #     assert is_instance(data_master_ids, list[int])
    #     assert is_instance(area_feature_ids, list[int])
    #
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         query = select(table.c.id).distinct().where(
    #             table.c['idata_master_fk'].in_(data_master_ids) &
    #             table.c['area_id'].in_(area_feature_ids)
    #         )
    #         self.log.debug(query)
    #         rlist = conn.execute(query).fetchall()
    #         return [result[0] for result in rlist]
    # # end

    # def _select_data_values_master_ids_by_plan(self, plan_name: str, area_feature_ids: list[int]) \
    #         -> tuple[list[int], list[int]]:
    #     assert is_instance(plan_name, str)
    #     assert is_instance(area_feature_ids, list[int])
    #
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         query = select(table.c.id, table.c['idata_master_fk']).distinct().where(
    #             (table.c['name'] == plan_name) &
    #             table.c['area_id'].in_(area_feature_ids)
    #         )
    #         self.log.debug(query)
    #         rlist = conn.execute(query).fetchall()
    #         data_master_ids = list({result[1] for result in rlist})
    #         plan_ids = list({result[0] for result in rlist})
    #         # data_values_master_ids -> plan_ids
    #
    #     if len(data_master_ids) > 1:
    #         self.log.warning(f"Multiple Data Masters for plan {plan_name}")
    #
    #     return plan_ids, data_master_ids[-1]
    # # end

    # -----------------------------------------------------------------------

    # def clear_predict_focussed_data(self, predict_master_id: int):
    #     # clear the content of 'tb_ipr_model_detail_focussed' and 'tb_ipr_train_data_focussed'
    #     with self.engine.connect() as conn:
    #         table = self.iPredictModelDetailFocussed
    #         query = delete(table).where(table.c['ipr_conf_master_id_fk'] == predict_master_id)
    #         self.log.debug(query)
    #         conn.execute(query)
    #
    #         table = self.iPredictTrainDataFocussed
    #         query = delete(table).where(table.c['ipr_conf_master_id_fk'] == predict_master_id)
    #         self.log.debug(query)
    #         conn.execute(query)
    #         conn.commit()

    # end
