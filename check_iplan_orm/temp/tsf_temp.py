
class IPredictTimeSeries(IPlanObject):

    def __init__(self, ipom, id: int, data_master_id: int):
        super().__init__(ipom)

        assert is_instance(id, int)
        assert is_instance(data_master_id, int)

        self._id: int = id
        self._plan: Optional[str] = None

        self._data_master: IDataMaster = self.ipom.data_masters().data_master(data_master_id)
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
    # Prediction plan
    # -----------------------------------------------------------------------

    @property
    def plan(self) -> Optional[IPredictionPlan]:
        return self._plan

    def set_plan(self, plan: str):
        assert is_instance(plan, str)
        self._plan = plan
        return self

    # -----------------------------------------------------------------------
    # Train/predict data
    # -----------------------------------------------------------------------

    def select_train_data(
        self,
        plan: Optional[str] = None,
        area: Union[None, int, list[int]] = None,
        skill: Union[None, int, list[int]] = None,
        new_format=True) -> DataFrame:
        """
        Retrieve the train data

        :param plan: name of the plan used for reference
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param new_format: DataFrame format
        :return: the dataframe used for training. It contains input/target features
        """

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])
        assert is_instance(plan, Optional[str])

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.plans().plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        df: DataFrame = self.ipom.select_train_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            new_format=new_format
        )
        return df
    # end

    def select_predict_data(
        self,
        plan: Optional[str] = None,
        area: Union[None, int, list[int], str, list[str]] = None,
        skill: Union[None, int, list[int], str, list[str]] = None,
        new_format=True) -> DataFrame:
        """
        Retrieve predict data

        :param plan:
        :param area:
        :param skill:
        :param new_format:
        :return:
        """
        assert is_instance(plan, Optional[str])
        assert is_instance(area, Union[None, int, list[int]])
        assert is_instance(skill, Union[None, int, list[int]])

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        df: DataFrame = self.ipom.select_predict_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
            new_format=new_format
        )
        return df
    # end

    def select_predict_data_ext(
        self,
        plan: Optional[str] = None,
        area: Union[None, int, list[int], str, list[str]] = None,
        skill: Union[None, int, list[int], str, list[str]] = None,
        #
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        periods: Optional[int] = None,
        freq: Literal['D', 'W', 'M'] = 'D',
        #
        new_format=True) -> DataFrame:
        """
        Retrieve predict data

        :param plan: name of the plan used for reference
        :param area: area(s) to select. If not specified, all available areas will be selected
        :param skill: skill(s) to select. If not specified, all available skills will be selected
        :param start_date: optional start date
        :param end_date: optional end date
        :param periods: n of periods. The period depends on the DataModel::PeriodHierarchy
        :param freq: period frequency
        :param new_format: DataFrame format
        :return: the dataframe used for prediction. It contains the input features only
        """
        assert is_instance(plan, Optional[str])
        assert is_instance(area, Union[None, int, list[int]])
        assert is_instance(skill, Union[None, int, list[int]])

        assert is_instance(start_date, Optional[datetime])
        assert is_instance(end_date, Optional[datetime])
        assert is_instance(periods, Optional[int])
        assert is_instance(freq, Literal['D', 'W', 'M'])

        area_feature_dict = self.area_hierarchy.feature_ids(leaf_only=False, with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(leaf_only=False, with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        input_feature_ids, _ = self.input_target_measure_ids
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        freq = self.period_hierarchy.freq if freq is None else freq
        periods = self.period_hierarchy.periods if periods is None else periods
        if start_date is not None and end_date is None:
            end_date = start_date + relativeperiods(periods=periods, freq=freq)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        df: DataFrame = self.ipom.select_predict_data_ext(
            data_master_id,
            plan_ids,
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
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        """
        assert is_instance(df, DataFrame)
        assert is_instance(plan, Optional[str])

        self.log.info("Save train data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        area_plan_map = pplan.area_plan_map()

        # if plan is not None:
        #     pplan = self.ipom.prediction_plan(plan, data_master_id)
        #     area_plan_map = pplan.area_plan_map()
        # else:
        #     area_plan_map = None

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
                    self.log.debugt(f"... [train] saving area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    # Note: type(*_feature_id) is a numpy type! converted into Python int
                    self.ipom.save_area_skill_train_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id), int(plan_id),
                        dfas, update=update)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to save train data for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return

    def delete_train_data(self,
                          plan: Optional[str] = None,
                          area: Union[None, int, list[int], str, list[str]] = None,
                          skill: Union[None, int, list[int], str, list[str]] = None,):
        """

        :param plan:
        :param area:
        :param skill:
        :return:
        """

        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])

        self.log.info("Deleting train data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, data_master_id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        self.ipom.delete_train_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
        )
        self.log.info("Done")
        return
    # end

    def delete_predict_data(self,
                            plan: Optional[str] = None,
                            area: Union[None, int, list[int]] = None,
                            skill: Union[None, int, list[int]] = None,):
        """

        :param plan:
        :param area:
        :param skill:
        :return:
        """
        assert is_instance(area, Union[None, int, list[int], str, list[str]])
        assert is_instance(skill, Union[None, int, list[int], str, list[str]])

        self.log.info("Deleting predict data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        pplan = self.ipom.prediction_plan(plan, self.data_master.id)
        if not pplan.exists():
            raise ValueError(f"Plan {plan} not existent")

        plan_ids = pplan.plan_ids

        # data_master_id: int,
        # data_values_master_ids: list[int],
        # area_feature_dict: dict[int, str],
        # skill_feature_dict: dict[int, str],
        # measure_dict: dict[int, str],
        # start_date: Optional[datetime] = None,
        # end_date: Optional[datetime] = None,
        # freq: Literal['D', 'W', 'M'] = 'D',
        # new_format=False):

        self.ipom.delete_predict_data(
            data_master_id,
            plan_ids,
            area_feature_dict,
            skill_feature_dict,
            measure_dict,
        )
        self.log.info("Done")
        return
    # end

    def save_predict_data(self, df: DataFrame,
                          plan: Optional[str] = None,
                          update: Optional[bool] = None):
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
                - None: all data is deleted and replaced
                        (delete and insert)
                - True: the data in the dataset replaces the same data in the database
                        (update or insert)
                - False:  all data in the database is not deleted or updated
                        (insert only)
        """
        assert is_instance(df, DataFrame)
        assert is_instance(plan, Optional[str])

        self.log.info("Save predict data ...")

        area_feature_dict = self.area_hierarchy.feature_ids(with_name=True)
        skill_feature_dict = self.skill_hierarchy.feature_ids(with_name=True)
        measure_dict = self.measure_ids(with_name=True)
        data_master_id = self.data_master.id
        plan = self._plan if plan is None else plan
        assert is_instance(plan, str)

        # 0) retrieve the Plan map
        #
        pplan = self.ipom.prediction_plan(plan, data_master_id)
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
                    self.log.debugt(f"... [pred] saving area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    # Note: type(*_feature_id) is a numpy type! converted into Python int
                    self.ipom.save_area_skill_predict_data(
                        int(area_feature_id), int(skill_feature_id), int(measure_id), int(plan_id),
                        dfas, update=update)
                except Exception as e:
                    exc = traceback.format_exc()
                    self.log.error(f"... unable to save predict data for area:{area_name}, skill:{skill_name}, measure:{measure_name}")
                    self.log.error(f"... {e}\n{exc}")
        # end
        self.log.info("Done")
        return
# end
