

tb_idata_values_master
tb_idata_values_detail
tb_idata_values_detail_hist
select * 
from tb_idata_values_detail tivd
limit 100
;


select * 
from tb_idata_values_detail tivd
limit 100
;

-- list of prediction plans available in 
-- tb_idata_values_detail AND tb_idata_values_detail_hist

select tivd.value_master_fk, count(1)
from tb_idata_values_detail tivd, tb_idata_values_detail_hist tivdh 
where tivd.value_master_fk = tivdh.value_master_fk 
group by tivd.value_master_fk
;

--13357278	3709776
--13357544	63504
--13357560	1524096
--13357572	1587600
--13357580	63504
--13357581	63504
--13357596	2709504


-- list of measures with a prediction plan

select distinct tivd.model_detail_id_fk 
from tb_idata_values_detail tivd
where tivd.value_master_fk  = 13357560
;

--13357278	29
--13357544	3 (5248, 5249, 5250)
--13357560	3 same as previous
--13357572	3 same as previous
--13357580	3 same as previous
--13357581	3 same as previous
--13357596	3 same as previous

select distinct tivdh.model_detail_id_fk 
from tb_idata_values_detail_hist tivdh
where tivdh.value_master_fk  = 13357560
;

--13357278	2  (4375, 4374)
--13357544	3 (5248, 5249, 5250)
--13357560	3 same as previous
--13357572	3 same as previous
--13357580	3 same as previous
--13357581	3 same as previous
--13357596	3 same as previous

-- Prediction plans with id in (13357560, 13357580, 13357572, 13357581, 13357596, 13357544)
-- retrieve the Data Master

select idata_master_fk , area_id  
from tb_idata_values_master tivm 
where tivm.id  in (13357560, 13357580, 13357572, 13357581, 13357596, 13357544)
;

-- data_master_id: 55
-- area_id (1005, 1021, 1034, 1042, 1043, 1058)

-- list of skills

select distinct tivdh.skill_id_fk  
from tb_idata_values_detail_hist tivdh
where tivdh.value_master_fk in  (13357560, 13357580, 13357572, 13357581, 13357596, 13357544)
order by tivdh.skill_id_fk
;

-- skill_ids: (1266 - 1310)

