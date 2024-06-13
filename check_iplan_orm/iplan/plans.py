from sqlalchemy import func

from .datamasters import *
import pandasx as pdx

# ---------------------------------------------------------------------------
# IDataValuesMaster == IPredictionPlans
# IPredictionPlan
# ---------------------------------------------------------------------------

class PredictionPlan(IPlanObject):

    def __init__(self, ipom, name: str, data_master: Union[int, str]):
        super().__init__(ipom)
        assert is_instance(name, str)
        assert is_instance(data_master, Union[None, int, str])
        self._name: str = name

        if data_master is not None:
            self._data_master = self.ipom.data_masters().data_master(data_master)
        else:
            self._data_master = None
        
        self.iDataValuesMaster = self.ipom.iDataValuesMaster
    # end

    @property
    def name(self) -> str:
        return self._name

    @property
    def data_master(self) -> DataMaster:
        return self._data_master

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        return self.data_master.area_hierarchy

    @property
    def plan_ids(self) -> list[int]:
        return self._select_plan_ids(
            self.name,
            [self.data_master.id],
            self.data_master.area_hierarchy.feature_ids(leaf_only=True)
        )

    # -----------------------------------------------------------------------
    # area_plan_map
    # -----------------------------------------------------------------------

    @property
    def area_plan_map(self) -> dict[int, int]:
        """
        Map area_id -> plan_id
        """
        plan_name = self.name
        data_master_id = self.data_master.id

        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(table.c['area_id', 'id']).where(
                (table.c['name'] == plan_name) &
                (table.c['idata_master_fk'] == data_master_id)
            )
            self.log.debug(query)
            pmap = {}
            rlist = conn.execute(query)
            for res in rlist:
                # area_id -> plan_id
                pmap[res[0]] = res[1]

        return pmap

    def _select_plan_ids(
        self,
        plan_name: Optional[str],
        data_master_ids: list[int],
        area_ids: list[int],
    ) -> list[int]:
        # alias: select_plan_ids(...)
        table = self.iDataValuesMaster
        if plan_name is None:
            query = select(table.c.id).where(
                table.c['idata_master_fk'].in_(data_master_ids) &
                table.c['area_id'].in_(area_ids)
            )
            self.log.debug(query)
        else:
            query = select(table.c.id).where(
                (table.c['name'] == plan_name) &
                table.c['idata_master_fk'].in_(data_master_ids) &
                table.c['area_id'].in_(area_ids)
            )
            self.log.debug(query)
        with self.engine.connect() as conn:
            rlist = conn.execute(query)
            return [result[0] for result in rlist]
    # end

    # -----------------------------------------------------------------------
    # data_range_map
    # -----------------------------------------------------------------------
    
    @property
    def date_range(self) -> tuple[datetime, datetime]:

        plan_name = self.name
        data_master_id = self.data_master.id
        
        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(table.c['area_id', 'start_date', 'end_date']).where(
                (table.c['name'] == plan_name) &
                (table.c['idata_master_fk'] == data_master_id)
            )
            self.log.debug(query)
            rlist = conn.execute(query)

            start_end_date_all = None
            start_end_date_map = {}
            all_equals = True

            for res in rlist:
                area_id, start_date, end_date = res

                # 'pd.datetime' IS NOT THE SAME than 'pd.date'
                start_date = pdx.to_datetime(start_date)
                end_date = pdx.to_datetime(end_date)

                start_end_date = start_date, end_date
                if start_end_date_all is None:
                    start_end_date_all = start_end_date
                elif start_end_date_all != start_end_date:
                    all_equals = False
                start_end_date_map[area_id] = start_end_date
        # end
        if all_equals:
            return start_end_date_all
        else:
            self.log.error(f"In plan {plan_name}, there are areas with different start_date/end_date pairs")
            return start_end_date_all
    # end

    # -----------------------------------------------------------------------
    # operations
    # -----------------------------------------------------------------------

    def exists(self) -> bool:
        name = self._name

        # data_master is optional

        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = select(func.count()).select_from(table).where(
                (table.c.name.like(f"%{name}%"))
            )
            if self._data_master is not None:
                query = query.where(
                    (table.c['idata_master_fk'] == self._data_master.id)
                )
            self.log.debug(query)
            count = conn.execute(query).scalar()
        return count > 0

    def delete(self):
        name = self._name

        # data_master is optional

        with self.engine.connect() as conn:
            table = self.iDataValuesMaster
            query = delete(table).where(
                (table.c.name.like(f"%{name}%"))
            )
            if self._data_master is not None:
                query = query.where(
                    (table.c['idata_master_fk'] == self._data_master.id)
                )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return self

    def create(self,
               start_date: datetime,
               end_date: Optional[datetime] = None,
               periods: Optional[int] = None,
               note: Optional[str] = None) -> "PredictionPlan":
        """
        Create a plan with the specified name for all areas in the area hierarchy
        If end_date or periods are not specified, it is used the PeriodHierarchy of the DataMaster

        :param start_date: start date
        :param end_date: end date
        :param periods: n of periods
        :return:
        """

        assert is_instance(start_date, datetime)
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])

        name = self._name

        freq = self.data_master.period_hierarchy.freq
        if periods is None or periods == 0:
            periods = self.data_master.period_hierarchy.periods
        if end_date is None:
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
        area_dict = self.area_hierarchy.feature_ids(with_name=True)
        now: datetime = datetime.now()
        note = "created by " + CREATED_BY if note is None else note
        count = 0
        with (self.engine.connect() as conn):
            table = self.iDataValuesMaster
            for area_id in area_dict:
                area_name = area_dict[area_id]
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
                    area_id=area_id,
                    last_updated_date=now,
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
        return f"{self.name}[{self._data_master.name}:{self._data_master.id}]"
# end


class PredictionPlans(IPlanObject):
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

    def __init__(self, ipom):
        super().__init__(ipom)

    # -----------------------------------------------------------------------

    def plan(self, name: Union[int, str, datetime], data_master: Union[None, int, str] = None) -> "PredictionPlan":
        assert is_instance(name, Union[int, str, datetime])
        assert is_instance(data_master, Union[None, int, str])

        plan_id: Union[int, str] = safe_int(name)

        if is_instance(name, datetime):
            now: datetime = datetime.now()
            now_str = now.strftime('%Y-%m-%d %H:%M:%S')
            name = f"Auto_Plan_OM_{now_str}"
        elif is_instance(plan_id, int):
            name, data_master = self._get_plan_name_and_data_master(plan_id)

        # WARN: it is not necessary to check for the data master.
        #       For example to delete ALL Plans with the same name

        # assert name is not None and data_master is not None, \
        #     "If the Plan is specified by name, it is required the Data_Master"

        return PredictionPlan(self.ipom, name, data_master)

    def _get_plan_name_and_data_master(self, plan_id: int) -> tuple[str, int]:
        assert is_instance(plan_id, int)

        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            query = select(table.c['name', 'idata_master_fk']).where(table.c.id == plan_id)
            self.log.debug(query)

            plan_name_data_master_id = conn.execute(query).fetchone()
            if plan_name_data_master_id is None:
                raise ValueError(f"Plan {plan_id} not found")

            return plan_name_data_master_id

    # -----------------------------------------------------------------------

    # def select_date_interval(self, id_or_name_or_date: Union[None, int, str, datetime] = None,
    #                          data_master_id: int = 0,
    #                          area_ids: Union[None, int, list[int]] = None) \
    #         -> Optional[tuple[datetime, datetime]]:
    #     """
    #     Retrieve the date interval used for the prediction based on several rules:
    #
    #         - prediction plan id
    #         - prediction plan name
    #         - date contained in the date interval
    #         - data_master_id
    #
    #     The prediction plan must be specific for a selected Data Master
    #     It is possible to specify the area(s). If the areas are not specified, it is
    #     selected the date interval as min(start_date), max(end_date) for all defined
    #     areas. If no plan is found, it is returned None
    #
    #     :param id_or_name_or_date: Prediction Plan id or name or date contained in the prediction interval
    #     :data_master_id: id of the Data Master to use
    #     :param area_ids: specific area(s) to consider
    #     :return: (start_date, end_date) OR None if no interval is found
    #     """
    #     assert is_instance(id_or_name_or_date, Union[None, int, str, datetime])
    #     assert is_instance(data_master_id, int)
    #     assert is_instance(area_ids, Union[None, int, list[int]])
    #
    #     area_ids: list[int] = as_list(area_ids)
    #     # convert a string representing an integer value into a integer
    #     id_or_name_or_date = safe_int(id_or_name_or_date)
    #
    #     if id_or_name_or_date is None:
    #         return self._select_by_data_master(data_master_id, area_ids)
    #     elif isinstance(id_or_name_or_date, datetime):
    #         start_end_date = self._select_by_date(id_or_name_or_date, data_master_id, area_ids)
    #     elif isinstance(id_or_name_or_date, int):
    #         start_end_date = self._select_by_id(id_or_name_or_date, data_master_id, area_ids)
    #     elif isinstance(id_or_name_or_date, str):
    #         start_end_date = self._select_by_name(id_or_name_or_date, data_master_id, area_ids)
    #     else:
    #         raise ValueError(f"Unsupported type for value {id_or_name_or_date}")
    #
    #     # (None, None) -> None
    #     if start_end_date[0] is None or start_end_date[1] is None:
    #         return None
    #     else:
    #         return start_end_date
    #
    # def _select_by_data_master(self, data_master_id: int, area_ids: list[int]) -> tuple[datetime, datetime]:
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         if len(area_ids) == 0:
    #             query = select(table.c['start_date', 'end_date']).where(
    #                 (table.c['idata_master_fk'] == data_master_id)
    #             )
    #         else:
    #             query = select(table.c['start_date', 'end_date']).where(
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c['area_id'].in_(area_ids))
    #             )
    #         self.log.debug(query)
    #         result = conn.execute(query).fetchone()
    #         return result[0], result[1]
    #
    # def _select_by_id(self, ppid: int, data_master_id: int, area_ids: list[int]) -> tuple[datetime, datetime]:
    #     # data_master_id & area_ids are not necessary BUT they are used to force consistency
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         if len(area_ids) == 0:
    #             query = select(table.c['start_date', 'end_date']).where(
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c.id == ppid)
    #             )
    #         else:
    #             query = select(table.c['start_date', 'end_date']).where(
    #                 (table.c.id == ppid) &
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c['area_id'].in_(area_ids))
    #             )
    #         self.log.debug(query)
    #         result = conn.execute(query).fetchone()
    #         return result[0], result[1]
    #
    # def _select_by_name(self, name: str, data_master_id: int, area_ids: list[int]) -> tuple[datetime, datetime]:
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         if len(area_ids) == 0:
    #             query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c['name'].like(f"%{name}%"))
    #             )
    #         else:
    #             query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c['name'].like(f"%{name}%")) &
    #                 (table.c['area_id'].in_(area_ids))
    #             )
    #         self.log.debug(query)
    #         result = conn.execute(query).fetchone()
    #         return (None, None) if result is None else result[0], result[1]
    #
    # def _select_by_date(self, when: datetime, data_master_id: int, area_ids: list[int]) -> tuple[datetime, datetime]:
    #     with self.engine.connect() as conn:
    #         table = self.iDataValuesMaster
    #         if len(area_ids) == 0:
    #             query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c['start_date'] <= when) &
    #                 (table.c['end_date'] >= when)
    #             )
    #         else:
    #             query = select(func.min(table.c['start_date']), func.max(table.c['end_date'])).where(
    #                 (table.c['idata_master_fk'] == data_master_id) &
    #                 (table.c['start_date'] <= when) &
    #                 (table.c['end_date'] >= when) &
    #                 (table.c['area_id'].in_(area_ids))
    #             )
    #         self.log.debug(query)
    #         result = conn.execute(query).fetchone()
    #         return result[0], result[1]
# end

