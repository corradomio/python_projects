Python interface to the iPlan Object Model


    read configuration   from 'ipr_conf_master_id'

    read historical data from  'tb_idata_values_detail_hist'

    create tables       'tb_ipr_model_detail_focussed'
                            id
                            ipr_conf_master_id_fk           <---
                            area_id_fk                      +
                            skill_id_fk                     +
                            best_model
                            best_model_name
                            best_r_2
                            best_wape
                            ohmodels_catftr
                        'tb_ipr_train_data_focussed'
                            ipr_conf_master_id_fk           <---
                            area_id_fk                      +
                            skill_id_fk                     +
                            target
                            actual
                            predicted
                            state_date

    delete the tables' content with the selected  ipr_conf_master_id_fk




Used with iPredict



    - create a connection to the DBMS

    - select the list of parameters specified in the

        select parameter_id
            from public.tb_ipr_conf_detail_focussed
            where ipr_conf_master_id = %ipredictConfMaster

    - select the ids of the leaves in Area Hierarchy used in the Data Master
        select area_id from tb_idata_values_master where idata_master_fk = %iDataId

    - select the historical data
        select state_date, skill_id_fk, area_id_fk, model_detail_id_fk, value
            from tb_idata_values_detail_hist
            where model_detail_id_fk in (-list or parameter_id extracted from the previous query-)
              and area_id_fk         in (-list of area leaf ids-)

    -


tb_idata_value_default
tb_idata_value_status
tb_idata_values_modulestatus
tb_idata_values_raw_map
tb_idata_values_raw_table
tb_idata_values_tag_detail
tb_idata_values_tag_master

tb_idata_values_master
tb_idata_values_detail
tb_idata_values_detail_hist


    iPredictMaster


    SELECT state_date, skill_id_fk, area_id_fk, model_detail_id_fk, value
      FROM tb_idata_values_detail_hist
     WHERE model_detail_id_fk IN (--list of measures--)
       AND  area_id_fk IN (--list of area  features--)
       AND (skill_id_fk IN (--list of skill features--)  OR skill_id_fk  IS NULL)


SELECT tb_idata_values_detail_hist.area_id_fk, tb_idata_values_detail_hist.skill_id_fk, tb_idata_values_detail_hist.model_detail_id_fk, tb_idata_values_detail_hist.state_date, tb_idata_values_detail_hist.value
FROM tb_idata_values_detail_hist
WHERE tb_idata_values_detail_hist.model_detail_id_fk IN (__[POSTCOMPILE_model_detail_id_fk_1])
  AND tb_idata_values_detail_hist.area_id_fk IN (__[POSTCOMPILE_area_id_fk_1])
  AND (tb_idata_values_detail_hist.skill_id_fk IN (__[POSTCOMPILE_skill_id_fk_1])
    OR tb_idata_values_detail_hist.skill_id_fk IS NULL)



What is 'tb_idata_value_master' ?
---------------------------------
    id                      // automatically generated
    start_date              // start date of the data
    end_date                // end date of the data
    name                    // dataset name
    created_date            // when the dataset was created
    idata_master_fk         -> [td_idata_master]
    loan_updated_time       ?? same date than 'created_date' ??
    isscenario              False
    temp_ind                False
    area_id                 -> [tb_attribute_detail] (area feature)
    last_updated_date       ?? same date than 'created_date' ??
    published               True/False
    published_id            ?? OPTIONAL: it has a value ONLY if 'published' is True

    note                    // generic text

    Is there an entry for EACH area feature?


How to fill 'tb_idata_values_detail_hist'
-----------------------------------------
    id                    // automatically generated
    value_master_fk       OPTIONAL -> [tb_idata_value_master]
    state_date            // value timestamp
    updated_date          // when the data was inserted in the database
    model_detail_id_fk    -> [tb_idata_model_detail] (measure)
    area_id_fk            -> [tb_attribute_detail]   (area feature)
    skill_id_fk           -> [tb_attribute_detail]   (skill feature)
    value                 // measure value
    value_type            // [NULL]
    value_insert_time     // [NULL]

