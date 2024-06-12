List of MINIMAL MANDATORY database updates

1) the tables

        tb_ipr_model_detail_focussed
        tb_ipr_train_data_focussed

    MUST BE ALREADY present


-----------------------------------------------------------------------------
tb_ipr_model_detail_focussed
-----------------------------------------------------------------------------

CREATE TABLE public.tb_ipr_model_detail_focussed (
	id bigserial NOT NULL,
	best_model text NULL,
	best_model_name text NULL,
	best_r_2 float8 NULL,
	best_wape float8 NULL,
	ohmodels_catftr text NULL,
	area_id_fk int8 NULL,
	ipr_conf_master_id_fk int8 NULL,
	skill_id_fk int8 NULL,
	CONSTRAINT tb_ipr_model_detail_focussed_pkey PRIMARY KEY (id)
);


-- public.tb_ipr_model_detail_focussed foreign keys

ALTER TABLE public.tb_ipr_model_detail_focussed ADD CONSTRAINT fk605wky4xax3agl31cmip5m86 FOREIGN KEY (area_id_fk) REFERENCES public.tb_attribute_detail(id);
ALTER TABLE public.tb_ipr_model_detail_focussed ADD CONSTRAINT fkeu6p4w89y31jcx4t2pi03rjj4 FOREIGN KEY (ipr_conf_master_id_fk) REFERENCES public.tb_ipr_conf_master_focussed(id);



-----------------------------------------------------------------------------
tb_ipr_test_prediction_values_detail_focussed
tb_ipr_predicted_values_detail_focussed
-----------------------------------------------------------------------------
DROP table public.tb_ipr_test_prediction_values_detail_focussed;


CREATE TABLE public.tb_ipr_test_prediction_values_detail_focussed (
	ipr_conf_master_id_fk int8 NULL,
	area_id_fk int8 NULL,
	skill_id_fk int8 NULL,
	model_detail_id_fk int8 NULL,
	actual numeric NULL,
	predicted numeric NULL,
	state_date date NULL
);

ALTER TABLE public.tb_ipr_test_prediction_values_detail_focussed ADD CONSTRAINT tb_ipr_test_prediction_values_detail_f_area_fk FOREIGN KEY (area_id_fk) REFERENCES public.tb_attribute_detail(id);
ALTER TABLE public.tb_ipr_test_prediction_values_detail_focussed ADD CONSTRAINT tb_ipr_test_prediction_values_detail_f_skill_fk FOREIGN KEY (skill_id_fk) REFERENCES public.tb_attribute_detail(id);
ALTER TABLE public.tb_ipr_test_prediction_values_detail_focussed ADD CONSTRAINT tb_ipr_test_prediction_values_detail_f_ipr_conf_master_f_fk FOREIGN KEY (ipr_conf_master_id_fk) REFERENCES public.tb_ipr_conf_master_focussed(id);
ALTER TABLE public.tb_ipr_test_prediction_values_detail_focussed ADD CONSTRAINT tb_ipr_test_prediction_values_detail_f_idata_model_detail_fk FOREIGN KEY (model_detail_id_fk) REFERENCES public.tb_idata_model_detail(id);


DROP table public.tb_ipr_predicted_values_detail_focussed;

CREATE TABLE public.tb_ipr_predicted_values_detail_focussed (
	ipr_conf_master_id_fk int8 NULL,
	area_id_fk int8 NULL,
	skill_id_fk int8 NULL,
	model_detail_id_fk int8 NULL,
	actual numeric NULL,
	predicted numeric NULL,
	state_date date NULL
);

ALTER TABLE public.tb_ipr_predicted_values_detail_focussed ADD CONSTRAINT tb_ipr_predicted_values_detail_f_area_fk FOREIGN KEY (area_id_fk) REFERENCES public.tb_attribute_detail(id);
ALTER TABLE public.tb_ipr_predicted_values_detail_focussed ADD CONSTRAINT tb_ipr_predicted_values_detail_f_skill_fk FOREIGN KEY (skill_id_fk) REFERENCES public.tb_attribute_detail(id);
ALTER TABLE public.tb_ipr_predicted_values_detail_focussed ADD CONSTRAINT tb_ipr_predicted_values_detail_f_idata_model_detail_id_fk FOREIGN KEY (model_detail_id_fk) REFERENCES public.tb_idata_model_detail(id);
ALTER TABLE public.tb_ipr_predicted_values_detail_focussed ADD CONSTRAINT tb_ipr_predicted_values_detail_f_iprconf_master_f_fk FOREIGN KEY (ipr_conf_master_id_fk) REFERENCES public.tb_ipr_conf_master_focussed(id);


-----------------------------------------------------------------------------
- INCONSISTENCIES
-----------------------------------------------------------------------------
E' UN BORDELLO!!!!!

    idata_model_details_id_fk/ model_detail_id_fk
    data_model_id_fk/ idatamodel_id_fk
    area_id_fk/area_id
.


There are SEVERAL inconsistencies in the name of the columns.

    tb_ipr_train_data_focussed.target -> model_detail_id_fk
