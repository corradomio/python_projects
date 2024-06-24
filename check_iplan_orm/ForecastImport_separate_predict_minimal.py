import pandas as pd
from sqlalchemy import create_engine, text as sql_text
import iPredict_17.TrainingPredictionRunner as tpr
import pickle

# load hyper_params, transformer and main model as pkl file
with open('dict_hp.pkl', 'rb') as f:
    dict_hp = pickle.load(f)

with open('dict_ohmodels_catFtr_areas.pkl', 'rb') as f:
    dict_ohmodels_catFtr_areas = pickle.load(f)

with open('dict_best_model_areas.pkl', 'rb') as f:
    dict_best_model_areas = pickle.load(f)

# Create prediction dataframe
inputTablePred = 'vw_food_import_pred'
connection = create_engine('postgresql://postgres:p0stgres@10.193.20.15:5432/adda')
cursor = connection.connect()
# query = "select * from "+inputTablePred
query = "select * from "+inputTablePred + " where item_country in ('CHICKEN MEAT - FROZEN~ARGENTINA', 'CHICKEN MEAT - FROZEN~BRAZIL') "
df_pred = pd.read_sql_query(sql_text(query), cursor)
connection.dispose()

#Process initial dates - prediction data
df_pred[dict_hp['dateCol']] = pd.to_datetime(df_pred[dict_hp['dateCol']])
df_pred[dict_hp['dowCol']] = df_pred[dict_hp['dateCol']].dt.day_name()
#df.sort_values(by=dateCol)

# now run pridiction for all areas
df_pred_out = tpr.run_prediction_by_area(df_pred, dict_hp, dict_ohmodels_catFtr_areas, dict_best_model_areas)

# create prediction dictionary by area for printing or saving to DB
dict_df_pred_area = dict(iter(df_pred_out.groupby(dict_hp['areaFeature'])))

