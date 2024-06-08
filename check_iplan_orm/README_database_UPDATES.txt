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
tb_ipr_train_data_focussed
-----------------------------------------------------------------------------

CREATE TABLE public.tb_ipr_train_data_focussed (
	ipr_conf_master_id_fk int8 NULL,
	area_id_fk int8 NULL,
	skill_id_fk int8 NULL,
	target int8 NULL,
	actual numeric NULL,
	predicted numeric NULL,
	state_date date NULL
);

ALTER TABLE tb_ipr_train_data_focussed ADD COLUMN model_detail_id_fk int8




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
