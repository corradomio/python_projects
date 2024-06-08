import traceback
from sqlalchemy import text
from stdlib.dict import reverse_dict
from .plans import *

# ---------------------------------------------------------------------------
# TimeSeriesFocussed Operations
#   Base class
# ---------------------------------------------------------------------------

class TSFOperations(IPlanObject):
    def __init__(self, tsf: "TimeSeriesFocussed"):
        super().__init__(tsf.ipom)
        self._tsf = tsf
        self._plan = self._tsf.plan

    # ----------------------------------------

    def delete(self):
        return self

    def save(self, *args, **kwargs):
        pass

    def select(self, **kwargs) -> DataFrame:
        pass
# end


# ---------------------------------------------------------------------------
# TrainFocussed
# TestFocussed
# PredictFocussed
# PredictedFocussed
# ---------------------------------------------------------------------------
# The implementation is a little 'strange':
#
#   class::operation()
#       .. collect all information necessary from the current oject ..
#       self._internal_op(information)
#
#   class::_internal_op(information)
#       .. implementation with minimum dependence with the current
#          object ..
#
# This for historical reasons: at the begin the '_internal_op()' methods
# were located in another class. They were removed and moved within the
# specialized class
#

class TrainFocussed(TSFOperations):
    def __init__(self, tsf: "TimeSeriesFocussed"):
        super().__init__(tsf.ipom)
        self._tsf = tsf
        self._plan = self._tsf.plan
    # end

    # -----------------------------------------------------------------------
    # delete
    # -----------------------------------------------------------------------

    def delete(self):
        data_master = self._tsf.data_master
        data_master_id = data_master.id
        plan_ids = self._plan.plan_ids

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)

        self._delete_train_data(
            data_master_id=data_master_id,
            plan_ids=plan_ids,
            area_feature_dict=area_feature_dict,
            skill_feature_dict=skill_feature_dict,
            measure_dict=measure_dict
        )
        return self

    def _delete_train_data(
            self, *,
            data_master_id: int,
            plan_ids: list[int],
            area_feature_dict: dict[int, str],
            skill_feature_dict: dict[int, str],
            measure_dict: dict[int, str]):

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
            table = self.ipom.iDataValuesDetailHist

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

    # -----------------------------------------------------------------------
    # save
    # -----------------------------------------------------------------------

    def save(self, df: DataFrame):
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
        """
        assert is_instance(df, DataFrame)

        self.log.info("Save train data ...")

        freq = self._tsf.freq
        area_plan_map = self._plan.area_plan_map()

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)

        # 1) normalize the dataframe
        #    columns: ['area_id_fk', 'skill_id_fk', 'state_date', <measure_id1>, ...]
        df = normalize_df(df, area_feature_dict, skill_feature_dict, measure_dict, freq)

        # 2) split df by area/skill (and drop the columns)
        dfdict = pdx.groups_split(df, groups=['area_id_fk', 'skill_id_fk'], drop=True)
        for area_skill in dfdict:
            area_feature_id: int = area_skill[0]
            skill_feature_id: int = area_skill[1]
            dfas = dfdict[area_skill]
            for measure_id in measure_dict.keys():
                area_name = area_feature_dict[area_feature_id]
                skill_name = skill_feature_dict[skill_feature_id]
                measure_name = measure_dict[measure_id]
                plan_id = None if area_plan_map is None or area_feature_id not in area_plan_map \
                    else area_plan_map[area_feature_id]
                try:
                    self.log.debugt(f"... [train] saving area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    # Note: type(*_feature_id) is a numpy type! converted into Python int
                    self._save_area_skill_train_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id), int(plan_id),
                        dfas)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to save train data for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return

    def _save_area_skill_train_data(
        self,
        area_feature_id: int, skill_feature_id: int, measure_id: int, plan_id: int,
        df: DataFrame):
        """

        :param area_feature_id:
        :param skill_feature_id:
        :param measure_id:
        :param plan_id:
        :param df:
        """

        assert is_instance(area_feature_id, int)
        assert is_instance(skill_feature_id, int)
        assert is_instance(measure_id, int)
        assert is_instance(df, DataFrame)
        assert is_instance(plan_id, int)

        # start_date = pdx.to_datetime(df['state_date'].min())
        # end_date = pdx.to_datetime(df['state_date'].max())
        n = len(df)

        now = datetime.now()

        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesDetailHist

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
    # end

    # -----------------------------------------------------------------------
    # select
    # -----------------------------------------------------------------------

    def select(
        self,
        area: Union[None, int, list[int], str, list[str]] = None,
        skill: Union[None, int, list[int], str, list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        periods: Optional[int] = None,
        new_format=True) -> DataFrame:
        """
        Retrieve the train data
        It is possible to specify a date range

        :param plan: name of the plan used for reference
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param start_date: optional start date
        :param end_date: optional end date
        :param periods: optional number of periods
        :param new_format: DataFrame format
        :return: the dataframe used for training. It contains input/target features
        """

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])
        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])

        # area & skill == 0 are converted in None
        if area == 0: area = None
        if skill == 0: skill = None

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.to_ids(area, with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.to_ids(skill, with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)
        data_master_id = self._tsf.data_master.id

        # plan, for training, is optional?
        plan_ids = self._plan.plan_ids if self._plan is not None else None

        # freq & periods are retrieved from the Data Master
        freq = self._tsf.data_master.period_hierarchy.freq
        if periods is None or periods <= 0:
            periods = self._tsf.data_master.period_hierarchy.periods

        start_date, end_date = start_end_dates(start_date, end_date, periods, freq)

        df: DataFrame = self._select_train_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            start_date, end_date,
            new_format=new_format
        )
        return df
    # end

    def _select_train_data(
        self,
        data_master_id: int,
        plan_ids: Optional[list[int]],  # data_values_master_ids
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        measure_dict: dict[int, str],
        start_date: Optional[datetime], end_date: Optional[datetime],
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
        :param start_date:
        :param end_date:
        :param new_format: if to create a dataframe compatible with
            the current implementation of the new format
        :return: a dataframe with the following columns
                if 'new_format == True':
                    columns: ['area:str', 'skill:str', 'date:datetime', <measure_1:float>, ...]
                else
                    columns: ['skill_id_fk:int', 'area_id_fk:int', 'time:datetime', 'day:str', <measure_1: float>, ...]
        """

        assert is_instance(data_master_id, int)
        assert is_instance(plan_ids, Optional[list[int]])
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(measure_dict, dict[int, str])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        table = self.ipom.iDataValuesDetailHist

        # 2) retrieve the data with 'skill NOT NULL'
        query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
            .where(table.c['model_detail_id_fk'].in_(measure_ids)
                   & table.c['area_id_fk'].in_(area_feature_ids)
                   & table.c['skill_id_fk'].in_(skill_feature_ids)
                   # & table.c['value_master_fk'].in_(plan_ids)
            )
        # 2.1) plan_ids is optional!
        if plan_ids is not None:
            query = query.where(table.c['value_master_fk'].in_(plan_ids))
        # 2.2) start/end dates are optional
        if start_date is not None and end_date is not None:
            query = query.where(
                (table.c['state_date'] >= start_date) &
                (table.c['state_date'] < end_date)
            )

        self.log.debug(query)
        df_with_skill = pd.read_sql_query(query, self.engine)

        # 3) retrieve the data with 'skill IS NULL'
        query = select(table.c['area_id_fk', 'skill_id_fk', 'model_detail_id_fk', 'state_date', 'value']) \
            .where(table.c['model_detail_id_fk'].in_(measure_ids) &
                   table.c['area_id_fk'].in_(area_feature_ids) &
                   (table.c['skill_id_fk'] == None)
            )
        # 3.1) plan_ids is optional!
        if plan_ids is not None:
            query = query.where(table.c['value_master_fk'].in_(plan_ids))
        # 3.2) start/end dates are optional
        if start_date is not None and end_date is not None:
            query = query.where(
                (table.c['state_date'] >= start_date) &
                (table.c['state_date'] < end_date)
            )

        self.log.debug(query)
        df_no_skill = pd.read_sql_query(query, self.engine)

        # 4) concatenate df_with_skill WITH df_no_skill
        df = concatenate_no_skill_df(df_with_skill, df_no_skill, skill_feature_dict)

        return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)
    # end

    # -----------------------------------------------------------------------
    # end
    # -----------------------------------------------------------------------
# end


class TestFocussed(TSFOperations):
    def __init__(self, tsf: "TimeSeriesFocussed"):
        super().__init__(tsf.ipom)
        self._tsf = tsf
        self._plan = self._tsf.plan

    # -----------------------------------------------------------------------
    # delete
    # -----------------------------------------------------------------------

    def delete(self):
        time_series_id = self._tsf.id

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)

        self._delete_test_data(
            time_series_id=time_series_id,
            area_feature_dict=area_feature_dict,
            skill_feature_dict=skill_feature_dict,
            measure_dict=measure_dict
        )
        return self

    def _delete_test_data(
        self, *,
        time_series_id: int,
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        measure_dict: dict[int, str]):
        assert is_instance(time_series_id, int)
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(measure_dict, dict[int, str])

        # 1) retrieve all area/skill feature ids
        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict)
        measure_ids = list(measure_dict.keys())

        # WARN: in theory it is not necessary to check for
        # ALL measures, but ONLY for the targets.
        # BUT in this way we can resolve also implementation's
        # errors!
        #
        # WARN: the colum 'target' has A WRONG name, it should be
        #
        #       model_detail_id_fk
        #

        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesDetailHist

            query = delete(table) \
                .where(
                (table.c['ipr_conf_master_id_fk'] == time_series_id) &
                table.c['area_id_fk'].in_(area_feature_ids) &
                table.c['skill_id_fk'].in_(skill_feature_ids) &
                table.c['target'].in_(measure_ids)
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return

    # -----------------------------------------------------------------------
    # save
    # -----------------------------------------------------------------------

    def save(self, test: DataFrame, predict: DataFrame):

        pass
# end


class PredictFocussed(TSFOperations):
    def __init__(self, tsf: "TimeSeriesFocussed"):
        super().__init__(tsf.ipom)
        self._tsf = tsf
        self._plan = self._tsf.plan

    # -----------------------------------------------------------------------
    # delete
    # -----------------------------------------------------------------------

    def delete(self):
        data_master = self._tsf.data_master
        data_master_id = data_master.id
        plan_ids = self._plan.plan_ids

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)

        self._delete_predict_data(
            data_master_id=data_master_id,
            plan_ids=plan_ids,
            area_feature_dict=area_feature_dict,
            skill_feature_dict=skill_feature_dict,
            measure_dict=measure_dict
        )
        return self

    def _delete_predict_data(
            self, *,
            data_master_id: int,
            plan_ids: list[int],
            area_feature_dict: dict[int, str],
            skill_feature_dict: dict[int, str],
            measure_dict: dict[int, str]):

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
            table = self.ipom.iDataValuesDetail

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

    # -----------------------------------------------------------------------
    # save
    # -----------------------------------------------------------------------

    def save(self, df: DataFrame):
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
        """
        assert is_instance(df, DataFrame)

        self.log.info("Save predict data ...")

        freq = self._tsf.freq
        area_plan_map = self._plan.area_plan_map()

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)

        # 1) normalize the dataframe
        #    columns: ['area_id_fk', 'skill_id_fk', 'state_date', <measure_id1>, ...]
        df = normalize_df(df, area_feature_dict, skill_feature_dict, measure_dict, freq)

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
                    self._save_area_skill_predict_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id), int(plan_id),
                        dfas)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(
                        f"... unable to save predict data for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return

    def _save_area_skill_predict_data(
        self,
        area_feature_id: int, skill_feature_id: int, measure_id: int, plan_id: int,
        df: DataFrame):
        """

        :param area_feature_id:
        :param skill_feature_id:
        :param measure_id:
        :param plan_id:
        :param df:
        """
        assert is_instance(area_feature_id, int)
        assert is_instance(skill_feature_id, int)
        assert is_instance(measure_id, int)
        assert is_instance(plan_id, int)
        assert is_instance(df, DataFrame)

        # start_date = pdx.to_datetime(df['state_date'].min())
        # end_date = pdx.to_datetime(df['state_date'].max())
        n = len(df)

        now = datetime.now()

        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesDetail

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
    # end

    # -----------------------------------------------------------------------
    # select
    # -----------------------------------------------------------------------

    def select(
        self,
        area: Union[None, int, list[int], str, list[str]] = None,
        skill: Union[None, int, list[int], str, list[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        periods: Optional[int] = None,
        new_format=True) -> DataFrame:
        """
        Retrieve predict data

        :param plan: name of the plan used for reference
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param start_date: optional start date
        :param end_date: optional end date
        :param periods: optional number of periods
        :param new_format: DataFrame format
        :return: the dataframe used for prediction. It contains input/target features
        """
        assert is_instance(area, Union[None, int, list[int]])
        assert is_instance(skill, Union[None, int, list[int]])
        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])

        # area & skill == 0 are converted in None
        if area == 0: area = None
        if skill == 0: skill = None

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.to_ids(area, with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.to_ids(skill, with_name=True)
        measure_dict: dict[int, str] = self._tsf.measure_ids(with_name=True)
        data_master_id = self._tsf.data_master.id
        plan_ids = self._plan.plan_ids

        # freq & periods are retrieved from the Data Master
        freq = self._tsf.data_master.period_hierarchy.freq
        if periods is None or periods <= 0:
            periods = self._tsf.data_master.period_hierarchy.periods

        start_date, end_date = start_end_dates(start_date, end_date, periods, freq)

        df: DataFrame = self._select_predict_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            start_date, end_date,
            new_format=new_format
        )
        return df
    # end

    def _select_predict_data(
        self,
        data_master_id: int,
        plan_ids: list[int],  # data_values_master_ids
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        measure_dict: dict[int, str],
        start_date: Optional[datetime], end_date: Optional[datetime],
        new_format=False) -> DataFrame:
        """

        :param data_master_id:
        :param plan_ids:
        :param area_feature_dict:
        :param skill_feature_dict:
        :param measure_dict:
        :param start_date:
        :param end_date:
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
        if start_date is not None and end_date is not None:
            qtext += """
                and tivd.state_date >= :start_date
                and tivd.state_date <  :end_date
                """
        query = text(qtext)
        self.log.debug(query)
        df = pd.read_sql_query(query, self.engine, params=dict(
            plan_ids=tuple(plan_ids),
            measure_ids=tuple(measure_ids),
            skill_feature_ids=tuple(skill_feature_ids),
            area_feature_ids=tuple(area_feature_ids),
            start_date=start_date,
            end_date=end_date
        ))

        return pivot_df(df, area_feature_dict, skill_feature_dict, measure_dict, new_format=new_format)
# end


class PredictedFocussed(TSFOperations):
    def __init__(self, tsf: "TimeSeriesFocussed"):
        super().__init__(tsf.ipom)
        self._tsf = tsf
        self._plan = self._tsf.plan

    # -----------------------------------------------------------------------
    # delete
    # -----------------------------------------------------------------------

    def delete(self):
        raise NotImplemented()
        return self

    # -----------------------------------------------------------------------
    # save
    # -----------------------------------------------------------------------

    def save(self, df: DataFrame):
        raise NotImplemented()
        return self

    # -----------------------------------------------------------------------
    # select
    # -----------------------------------------------------------------------

    def select(self):
        raise NotImplemented()
        return self
# end


# ---------------------------------------------------------------------------
# ModelsFocussed
# ---------------------------------------------------------------------------

class ModelsFocussed(TSFOperations):
    def __init__(self, tsf: "TimeSeriesFocussed"):
        super().__init__(tsf.ipom)
        self._tsf = tsf
        self._plan = self._tsf.plan

    # -----------------------------------------------------------------------
    # delete
    # -----------------------------------------------------------------------

    def delete(self):
        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)

        self._delete_models(
            time_series_id=self._tsf.id,
            area_feature_dict=area_feature_dict,
            skill_feature_dict=skill_feature_dict,
        )
        return self

    def _delete_models(
        self, *,
        time_series_id: int,
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str]
    ):
        assert is_instance(time_series_id, int)
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])

        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict.keys())

        with self.engine.connect() as conn:
            table = self.ipom.iPredictModelDetailFocussed
            query = delete(table).where(
                (table.c['ipr_conf_master_id_fk'] == time_series_id) &
                table.c['area_id_fk'].in_(area_feature_ids) &
                table.c['skill_id_fk'].in_(skill_feature_ids)
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        # end
    # end

    # -----------------------------------------------------------------------
    # save
    # -----------------------------------------------------------------------
    # List of models:
    #
    #   {
    #       (area, skill) : {
    #           ???
    #       }
    #   }

    def save(self, models: dict[
            Union[tuple[int, int], tuple[str, str]],
            dict]):

        assert is_instance(models, dict[Union[tuple[int, int], tuple[str, str]], dict])

        if len(models) == 0:
            return self

        area_skill_key = list(models.keys())[0]
        new_format = is_instance(area_skill_key, tuple[str, str])

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)

        self._save_models(
            models=models,
            area_feature_dict=area_feature_dict,
            skill_feature_dict=skill_feature_dict,
            new_format=new_format)
        return self

    def _save_models(
        self, *,
        models: dict[Union[tuple[int, int], tuple[str, str]], dict],
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        new_format: bool):

        area_feature_drev = reverse_dict(area_feature_dict)
        skill_feature_drev = reverse_dict(skill_feature_dict)

        bulk_data = []

        for area_skill_key in models:
            if new_format:
                area, skill = area_skill_key
                area_id = area_feature_drev[area]
                skill_id = skill_feature_drev[skill]
            else:
                area_id, skill_id = area_skill_key
            # end

            model_info = models[area_skill_key]

            bulk_data.append(dict(
                area_id_fk=area_id,
                skill_id_fk=skill_id,
                best_model_name=model_info['name'],
                best_model=model_info['model'],
                ohmodels_catftr=model_info['ohmodels_catftr'],
                best_r_2=model_info['r2'],
                best_wape=model_info['wape'],
            ))

        with self.engine.connect() as conn:
            table = self.ipom.iPredictModelDetailFocussed
            conn.execute(table.insert(), bulk_data)
            conn.commit()
        pass
    # end

    # -----------------------------------------------------------------------
    # select
    # -----------------------------------------------------------------------
    # {
    #    (area_id, skill_id)
    # }
    # new_format: area & skill in string format

    def select(self, new_format=False) \
        -> dict[
            Union[tuple[int, int], tuple[str, str]],
            dict]:
        assert is_instance(new_format, bool)

        area_feature_dict: dict[int, str] = self._tsf.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict: dict[int, str] = self._tsf.skill_hierarchy.feature_ids(with_name=True)

        self._select_models(
            time_series_id=self._tsf.id,
            area_feature_dict=area_feature_dict,
            skill_feature_dict=skill_feature_dict,
        )
        return self

    def _select_models(
        self, *,
        time_series_id: int,
        area_feature_dict: dict[int, str],
        skill_feature_dict: dict[int, str],
        new_format: bool
    ) -> dict[
            Union[tuple[int, int], tuple[str, str]],
        dict]:
        assert is_instance(time_series_id, int)
        assert is_instance(area_feature_dict, dict[int, str])
        assert is_instance(skill_feature_dict, dict[int, str])
        assert is_instance(new_format, bool)

        area_feature_ids = list(area_feature_dict.keys())
        skill_feature_ids = list(skill_feature_dict.keys())

        area_feature_drev = reverse_dict(area_feature_dict)
        skill_feature_drev = reverse_dict(skill_feature_dict)

        models: dict[tuple[int, int], dict] = {}

        with (self.engine.connect() as conn):
            table = self.ipom.iPredictModelDetailFocussed
            query = select(table.c[
                               'area_id_fk', 'skill_id_fk',
                               'best_model_name', 'best_model', 'ohmodels_catftr',
                               'best_r_2', 'best_wape'
                           ]).where(
                (table.c['ipr_conf_master_id_fk'] == time_series_id) &
                table.c['area_id_fk'].in_(area_feature_ids) &
                table.c['skill_id_fk'].in_(skill_feature_ids)
            )
            self.log.debug(query)
            for res in conn.execute(query):
                area_id, skill_id, name, model, ohmodels_catftr, r2, wape = res

                # new_format: area/skill as strings
                area_skill_key: Union[tuple[int, int], tuple[str, str]] = \
                    (area_feature_drev[area_id], skill_feature_drev[skill_id]) if new_format else (area_id, skill_id)

                models[area_skill_key] = dict(
                    name=name,
                    model=model,
                    ohmodels_catftr=ohmodels_catftr,
                    r2=r2,
                    wape=wape
                )
        # end
        return models
    # end
# end


# ---------------------------------------------------------------------------
# IPlanTimeSeries
#   TimeSeriesFocussed
# ---------------------------------------------------------------------------

class TSFParameter(IPlanData):
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
    def measure(self) -> Measure:
        """
        Retrieve the measure containing the data
        the predicted data
        """

        self.check_data()
        measure_id = self.data['parameter_id']
        return None if measure_id is None else Measure(self.ipom, measure_id)

    @property
    def populate(self) -> Optional[Measure]:
        """
        Retrieve the measure where to save the data
        """

        self.check_data()
        measure_id = self.data['to_populate']
        return None if measure_id is None else Measure(self.ipom, measure_id)

    def __repr__(self):
        return f"{self.name}[{self.id}, {self.measure.name}:{'out' if self.is_target else 'in'}]"
# end


class TimeSeriesFocussed(IPlanData):
    def __init__(self, ipom, id):
        super().__init__(ipom, id, ipom.iPredictMasterFocussed)
        self._plan: Optional[PredictionPlan] = None
        self._data_master: Optional[DataMaster] = None
        self._parameters: list[TSFParameter] = []

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name

        self.check_data()
        return self._data['ipr_conf_master_name']

    @property
    def description(self) -> str:
        self.check_data()
        return self._data['ipr_conf_master_desc']

    @property
    def data_model(self) -> DataModel:
        self.check_data()
        data_model_id = self._data['idata_model_details_id_fk']
        return self.ipom.data_models().data_model(data_model_id)

    @property
    def area_hierarchy(self) -> AttributeHierarchy:
        self.check_data()
        area_hierarchy_id = self._data['area_id_fk']
        return self.ipom.hierachies().area_hierarchy(area_hierarchy_id)

    @property
    def skill_hierarchy(self) -> AttributeHierarchy:
        self.check_data()
        skill_hierarchy_id = self._data['skill_id_fk']
        return self.ipom.hierachies().skill_hierarchy(skill_hierarchy_id)

    @property
    def data_master(self, find=False) -> Optional[DataMaster]:
        if self._data_master is not None:
            return self._data_master

        # check if the time series is published
        self.check_data()
        data_master_id = self._data['idata_id_fk']
        if data_master_id is not None:
            self._data_master = self.ipom.data_masters().data_master(data_master_id)
            return self._data_master

        # the time series is not published BUT there exists a SINGLE
        #       DataMaster = (DataMode, AreaHierarchy, SkillHierarcy)
        if find:
            self._data_master = self.ipom.data_masters().find_data_master(
                self.data_model.id,
                self.area_hierarchy.id,
                self.skill_hierarchy.id
            )

        return self._data_master

    # @property
    # def period_hierarchy(self) -> PeriodHierarchy:
    #     return self.data_master.period_hierarchy

    @property
    def freq(self) -> Literal['D', 'W', 'M']:
        return self.data_master.period_hierarchy.freq

    # -----------------------------------------------------------------------
    # Time Series parameters/measures
    # -----------------------------------------------------------------------

    @property
    def parameters(self) -> list[TSFParameter]:
        if len(self._parameters) > 0:
            return self._parameters

        tsf_id = self._id
        skill_hierarchy_id = self.skill_hierarchy.id
        with self.engine.connect() as conn:
            table = self.ipom.iPredictDetailFocussed
            query = select(table).where(
                (table.c['ipr_conf_master_id'] == tsf_id) &
                (table.c['skill_id_fk'] == skill_hierarchy_id)
            )
            self.log.debug(f"{query}")
            rlist = conn.execute(query).fetchall()
            # idlist: [(id,), ...]
        self._parameters = [TSFParameter(self.ipom, to_data(result)) for result in rlist]
        return self._parameters

    @property
    def measures(self) -> tuple[list[Measure], list[Measure]]:
        """
        Retrieve the list of measures used as 'target' and 'input features'

        :return: ([<target measure>, ...], [<input feature measure>, ...])
        """
        tsf_id = self._id
        skill_hierarchy_id = self.skill_hierarchy.id
        inputs: list[Measure] = []
        targets: list[Measure] = []
        with self.engine.connect() as conn:
            table = self.ipom.iPredictDetailFocussed
            query = select(table.c['parameter_id', 'parameter_value', 'to_populate']) \
                .where(
                    (table.c['ipr_conf_master_id'] == tsf_id) &
                    (table.c['skill_id_fk'] == skill_hierarchy_id)
                )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            for res in rlist:
                measure_id, parameter_type, populate_id = res
                if parameter_type == 'input':
                    inputs.append(Measure(self.ipom, measure_id))
                elif parameter_type == 'output':
                    targets.append(Measure(self.ipom, measure_id))
                    # populate
                else:
                    self.log.error(f"Unsupported parameter type {parameter_type}")
        # end
        return targets, inputs

    def parameter(self, measure: Union[None, int, str]) -> TSFParameter:
        """
        Retrieve the parameter related to the specified measure
        :param measure: with measure (as id or name)
        :return:
        """
        is_name = is_instance(measure, str)

        for p in self.parameters:
            m = p.measure
            if is_name and m.name == measure or not is_name and m.id == measure:
                return p

        raise ValueError(f"No parameters assigned to measure {measure}")

    def measure_ids(self, with_name=False) -> Union[list[int], dict[int, str]]:
        """
        Retrieve the list of measures used as 'target' and 'input features'

        :return: ([<target measure>, ...], [<input feature measure>, ...])
        """
        assert is_instance(with_name, bool)
        if with_name:
            target_dict, input_dict = self._measure_dict()
            return target_dict | input_dict
        else:
            target_list, input_list = self._measure_ids()
            return target_list + input_list

    def target_input_ids(self, with_name=False) \
        -> Union[
            tuple[list[int], list[int]],
            tuple[dict[int, str], dict[int, str]]
        ]:
        assert is_instance(with_name, bool)
        if with_name:
            return self._measure_dict()
        else:
            return self._measure_ids()

    def _measure_dict(self) -> tuple[dict[int, str], dict[int, str]]:
        tsf_id = self._id
        skill_feature_ids: list[int] = self.skill_hierarchy.feature_ids()
        input_dict: dict[int, str] = {}
        target_dict: dict[int, str] = {}
        with self.engine.connect() as conn:
            query = text("""
                select ticd.parameter_id measure_id, ticd.parameter_value parameter_type, timd.measure_id measure_name
                  from tb_ipr_conf_detail_focussed ticd,
                       tb_idata_model_detail timd
                 where ticd.ipr_conf_master_id = :tsf_id
                   and ticd.skill_id_fk in :skill_feature_ids
                   and ticd.parameter_id = timd.id
            """)
            self.log.debug(query)
            rlist = conn.execute(query, parameters=dict(
                tsf_id=tsf_id,
                skill_feature_ids=tuple(skill_feature_ids)
            )).fetchall()
            for res in rlist:
                measure_id, parameter_type, measure_name = res
                if parameter_type == 'input':
                    input_dict[measure_id] = measure_name
                elif parameter_type == 'output':
                    target_dict[measure_id] = measure_name
                else:
                    self.log.error(f"Unsupported parameter type {parameter_type}")
        return target_dict, input_dict

    def _measure_ids(self) -> tuple[list[int], list[int]]:
        tsf_id = self._id
        skill_hierarchy_id = self.skill_hierarchy.id
        input_ids: list[int] = []
        target_ids: list[int] = []
        with self.engine.connect() as conn:
            table = self.ipom.iPredictDetailFocussed
            query = select(table.c['parameter_id', 'parameter_value', 'to_populate']) \
                .where(
                (table.c['ipr_conf_master_id'] == tsf_id) &
                (table.c['skill_id_fk'] == skill_hierarchy_id)
            )
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()

            for res in rlist:
                measure_id, parameter_type, populate_id = res
                if parameter_type == 'input':
                    input_ids.append(measure_id)
                elif parameter_type == 'output':
                    target_ids.append(measure_id)
                    # populate
                else:
                    self.log.error(f"Unsupported parameter type {parameter_type}")
        return target_ids, input_ids

    # -----------------------------------------------------------------------
    # plan
    # -----------------------------------------------------------------------
    # There are 3 possible Data Master
    #   assigned to the time series
    #   assigned to the Plan
    #   passed as argument
    # It has sense ???
    # -----------------------------------------------------------------------

    @property
    def plan(self) -> Optional[PredictionPlan]:
        return self._plan

    def set_plan(self, plan: str, data_master: Union[None, int, str] = None):
        assert is_instance(plan, str)
        assert is_instance(data_master, Union[None, int, str])

        # if data_master is not specified, try the Data Master assigned to the
        # Time Series, IF it is present
        if data_master is None:
            if self.data_master is not None:
                data_master = self.data_master.id

        # if data_master is not specified yet, try the Data Master assigned to the
        # Plan, IF it is present and UNIQUE
        if data_master is None:
            data_master = self._find_plan_data_master_id(plan)

        self._plan = self.ipom.plans().plan(plan, data_master)
        assert self._plan.exists(), f"Unknown plan {plan}"

        # is the TS Data Master is None, it is used the Plan's Data Matsre
        if self._data_master is None:
            self._data_master = self._plan.data_master

        self_data_master = self.data_master
        plan_data_master = self._plan.data_master

        assert plan_data_master.area_hierarchy.id == self.area_hierarchy.id and \
            plan_data_master.skill_hierarchy.id == self.skill_hierarchy.id and \
            plan_data_master.data_model.id == self.data_model.id, \
            (f"Inconsistent Data Master {data_master} with Time Series "
             f"({self.data_model.name}, {self.area_hierarchy.name}, {self.skill_hierarchy.name},)")

        assert self_data_master.id == plan_data_master.id, \
            f"Inconsistent Plan Data Master '{plan_data_master.name}' with Time Series Data Master '{self_data_master.name}'"

        return self

    # alias_of(set_plan)
    def using_plan(self, plan: str, data_master: Union[None, int, str] = None):
        return self.set_plan(plan, data_master)

    def _find_plan_data_master_id(self, plan: str) -> Optional[int]:
        with self.engine.connect() as conn:
            table = self.ipom.iDataValuesMaster
            query = select(table.c['idata_master_fk'], func.count(table.c['idata_master_fk'])) \
                .where(table.c['name'] == plan) \
                .group_by(table.c['idata_master_fk'])
            self.log.debug(query)
            rlist = conn.execute(query).fetchall()
            if len(rlist) > 1:
                raise ValueError(f"Invalid plan '{plan}': assigned to {len(rlist)} Data Masters")
            elif len(rlist) == 0:
                raise ValueError(f"Invalid plan '{plan}': no Data Masters assigned")

        data_master_id = rlist[0][0]
        return data_master_id

    # -----------------------------------------------------------------------

    def set_data_master(self, data_master: Union[int, str]):
        assert is_instance(data_master,  Union[int, str])
        self._data_master = self.ipom.data_masters().data_master(data_master)
        return self

    # alias_of(set_data_master)
    def using_data_master(self, data_master: Union[int, str]):
        return self.set_data_master(data_master)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    def train(self) -> TrainFocussed:
        """
        TS Manager used to save the train data
        """
        return TrainFocussed(self)

    def predict(self) -> PredictFocussed:
        """
        TS Manager used to save the prediction input features
        """
        return PredictFocussed(self)

    def predicted(self) -> PredictedFocussed:
        """
        TS Manager used to save the predicted values
        """
        return PredictedFocussed(self)

    def test(self) -> TestFocussed:
        """
        TS Magaser used to save the test data: (actual/predicted)
        """
        return TestFocussed()

    def models(self) -> ModelsFocussed:
        """
        TS Manager used to save the trained models
        """
        return ModelsFocussed(self)

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def delete(self):
        """
        Delete the time series
        :return:
        """
        if self._id == NO_ID:
            return self

        # delete all train/predict data
        self.train().delete()
        self.predict().delete()
        self.models().delete()
        self.predicted().delete()

        # delete the TS definition
        self._name = self.name
        self._delete_time_series_focussed(self._id)
        super().delete()
        return self

    def _delete_time_series_focussed(self, tsf_id: int):
        with self.engine.connect() as conn:
            # 1) delete details
            table = self.ipom.iPredictDetailFocussed
            query = delete(table).where(
                table.c['ipr_conf_master_id'] == tsf_id
            )
            self.log.debug(query)
            conn.execute(query)

            # 2) delete master
            table = self.ipom.iPredictMasterFocussed
            query = delete(table).where(
                table.c.id == tsf_id
            )
            self.log.debug(query)
            conn.execute(query)
            conn.commit()
        return

    def create(
        self, *,
        targets: Union[str, list[str]],
        populate: Union[None, str, list[str]] = None,
        inputs: Union[None, str, list[str]] = None,
        data_master: Union[None, int, str] = None,
        description: Optional[str] = None):
        assert is_instance(targets, Union[str, list[str]])
        assert is_instance(populate, Union[None, str, list[str]])
        assert is_instance(inputs, Union[None, str, list[str]])
        assert is_instance(data_master, Union[None, int, str])
        assert is_instance(description, Optional[str])
        """
        Create the TS.
        Note 
        """

        if self._id != NO_ID:
            self.log.warning(f"Time Series '{self._name}' already existent")
            return self

        targets = as_list(targets)
        populate = as_list(populate)
        inputs = as_list(inputs)

        assert is_instance(self._name, str)

        self._id = self._create_time_series_focussed(
            self._name,
            targets=targets,
            populate=populate,
            inputs=inputs,
            data_master=data_master,
            description=description
        )
        return self

    def _create_time_series_focussed(
        self, name: str, *,
        targets: Union[str, list[str]],
        populate: Union[None, str, list[str]],
        inputs: Union[None, str, list[str]] = None,
        data_master: Union[None, int, str] = None,
        description: Optional[str] = None) -> int:
        """
        Create a time series

        :param name: Time Series name
        :param targets: list of target measures
        :param populate: list of target measures used to save the predicted values
        :param inputs: list of input measures
        :param data_master: Data Master
        :param description: description
        """

        assert is_instance(name, str)
        assert is_instance(targets, Union[str, list[str]])
        assert is_instance(populate, Union[str, list[str]])
        assert is_instance(inputs, Union[None, str, list[str]])
        assert is_instance(data_master, Union[None, int, str])
        assert is_instance(description, Optional[str])

        data_master = self.ipom.data_masters().data_master(data_master)
        data_model: DataModel = data_master.data_model
        data_model_id = data_model.id
        area_hierarchy_id = data_master.area_hierarchy.id
        skill_hierarchy_id = data_master.skill_hierarchy.id

        targets = as_list(targets, 'targets')
        populate = as_list(populate, 'populate')
        inputs = as_list(inputs, 'inputs')
        description = name if description is None else description
        n_targets = len(targets)

        # populate can be the empty list OR a list with tha same length
        # of the targets
        assert len(populate) == 0 or len(populate) == n_targets

        populate_list: list[Union[None, int, str]] = []
        if len(populate) == 0:
            populate_list = [None]*n_targets
        else:
            populate_list = populate

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
            self.log.debug(query)
            tsf_id = conn.execute(query).scalar()
            # 2) fill tb_ipr_conf_detail_focussed
            table = self.ipom.iPredictDetailFocussed
            for i in range(n_targets):
                measure = targets[i]
                measure_id = data_model.measure(measure).id

                topop = populate_list[i]
                topop_id = None if topop is None else data_model.measure(topop).id

                # NOT SUPPORTED yet
                period = None

                query = insert(table).values(
                    parameter_desc=measure,
                    parameter_value='output',
                    ipr_conf_master_id=tsf_id,
                    parameter_id=measure_id,
                    skill_id_fk=skill_hierarchy_id,
                    to_populate=topop_id,
                    period=period
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

    # -----------------------------------------------------------------------
    # End
    # -----------------------------------------------------------------------
# end


class IPlanTimeSeries(IPlanObject):

    def __init__(self, ipom):
        super().__init__(ipom)

    def focussed(self, id: Union[int, str]) -> TimeSeriesFocussed:
        tsf_id = self._convert_id(
            id, self.ipom.iPredictMasterFocussed, ['ipr_conf_master_name', 'ipr_conf_master_desc'], nullable=True)

        if tsf_id is None:
            return TimeSeriesFocussed(self.ipom, id)
        else:
            return TimeSeriesFocussed(self.ipom, tsf_id)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
