INCONSISTENCIES between
    idata_model_details_id_fk/ model_detail_id_fk
    data_model_id_fk / idatamodel_id_fk
    area_id_fk / area_id


tb_ipr_train_data_focussed
    target
        WRONG NAME! it should be 'model_detail_id_fk'

tb_idata_model_detail
    meaasure_id
        WRONG NAME: is it a measure name!


tb_ipr_conf_master_focussed
    idata_model_details_id_fk -> tb_idata_model_master
        WRONG column name! It is a reference to 'tb_idata_model_detail'