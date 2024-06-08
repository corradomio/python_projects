JSON parameters:
    "id"              ipredict_conf_master_id (ts_id)
    "iDataId"         data_master_id
    "planId"          plan_id
    "areaFeature"     area_fature_id
    "skillFeature"    skill_feature_id


training:
    "id": "68",
    "iDataId": "48",
    "areaFeature": 0,
    "skillFeature": 0,
    "planId": 0,


prediction:
    "id": "68",
    "areaFeature": 953,
    "skillFeature": 993,
    "planId": 13356658,
    "planDate": "Apr 29, 2024",


current:
    "id": "68",
    "areaFeature": 0,
    "skillFeature": 0,
    "planId": 0,



THEN:
    the TS has references to (Data Model, Area/Skill Hierarchy)
    it is passed Data Master's id -> it is necessary to check the CONSISTENCY
        BETWEEN TS's (Data Model, Area/Skill Hierarchy)
            AND Data Master's (Data Model, Area/Skill Hierarchy)


-----------------------------------------------------------------------------
# Used in iPredict Server
-----------------------------------------------------------------------------

tb_ipr_conf_master_focussed     [TimeSeries]
    id
    ipr_conf_master_name
    ipr_conf_master_desc
    idata_model_details_id_fk
    area_id_fk
    skill_id_fk
    idata_id_fk

tb_ipr_conf_detail_focussed     [TimeSeries details: in/out measures]
    id
    parameter_desc
    parameter_value
    ipr_conf_master_id
    parameter_id
    to_populate
    skill_id_fk
    period


-----------------------------------------------------------------------------
# Used in iPredict Server
-----------------------------------------------------------------------------

tb_idata_values_master      [Plan]
    id
    start_date
    end_date
    name
    created_date
    idata_master_fk
    loan_updated_time
    published
    isscenario
    temp_ind
    area_id
    last_updated_date
    published_id
    note


tb_idata_values_detail_hist [Training Data]
    id
    value_master_fk
    state_date
    updated_date
    model_detail_id_fk
    skill_id_fk
    value
    ----------------------
    value_type
    value_insert_time
    area_id_fk


tb_idata_values_detail      [Prediction Data]
    id
    value_master_fk
    state_date
    updated_date
    model_detail_id_fk
    skill_id_fk
    value
    ----------------------  WHY these columns are missing???
    ?value_type
    ?value_insert_time
    ?area_id_fk


-----------------------------------------------------------------------------
# Not used in iPredict Server
-----------------------------------------------------------------------------

tb_ipr_data_master
    id
    data_master_name
    description


tb_ipr_data_detail
    id
    data_master_id
    db_detail_id
    train_tb
    pred_tb
    col_map_id

tb_ipr_master
    id
    description
    area_hierarchy_fk
    ipr_conf_id
    ipr_data_id
    skill_hierarchy_fk
    idatamaster_id_fk


-----------------------------------------------------------------------------
# Python ML models Models
-----------------------------------------------------------------------------
#  tb_ipr_conf_detail
#  tb_ipr_conf_master

#  tb_ipr_data_detail
#  tb_ipr_data_master
#  tb_ipr_master


#
# where to save the models
#
tb_ipr_model_detail_focussed
    id                      seq
    best_model              JSON ???
    best_model_name         model name
    best_r_2                r2 score
    best_wape               wape score
    ohmodels_catftr         ???
    area_id_fk              area feature
    skill_id_fk             skill feature
    ipr_conf_master_id_fk   time series


#
# where to save the test data & predicted data
#
tb_ipr_train_data_focussed
    ipr_conf_master_id_fk
    area_id_fk
    skill_id_fk
    target
    actual
    predicted
    state_date

