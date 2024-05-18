A iPlan timeseries is composed by 2 parts:

    1) the TB definition
    2) the Data Master used to retrieve the data
    3) the Plan used to retrieve the start/end date for the prediction

The training is specified by:
    ts_id (tb_ipr_conf_master_focussed/tb_ipr_conf_detail_focussed)
    data_master_id (tb_idata_master)

The prediction is specified by
    ts_id (ipredict_configuration_master_id)
    plan_id (tp_idata_values_master), but, from Plan, it is possible to retrieve
        start/end date
        data_master_id

