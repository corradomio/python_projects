import pandas as pd
from sqlalchemy import create_engine, text as sql_text
import matplotlib.pyplot as plt
import iPredict_17.TrainingPredictionRunner as tpr
import pickle


def create_custom_regression_models():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import ElasticNetCV
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import linear_model
    from sklearn.ensemble import GradientBoostingRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    list_regression_models ={'BaysR':linear_model.BayesianRidge(),
                             'Lasso':linear_model.Lasso(alpha=0.1),
                             'DTR':DecisionTreeRegressor(random_state=0),
                             'LinR':LinearRegression(),
                             #'LogR':LogisticRegression(),
                             #'LogR_c200pl2':LogisticRegression(C=100, penalty="l2"),
                             'MLP':MLPRegressor(max_iter=1000),
                             'MLP_hl33lr0.01es':MLPRegressor(hidden_layer_sizes=(3, 3), learning_rate_init=0.01, early_stopping=True, max_iter=1000),
                             'SVR':SVR(),
                             'SVR_krRBFc0.1e0.9':SVR(kernel="rbf",C=0.1,epsilon=0.9),
                             #'SVR_krLinc0.1e0.9':SVR(kernel="linear",C=0.1,epsilon=0.9),
                             'SVR_krSigc0.1e0.9':SVR(kernel="sigmoid",C=0.1,epsilon=0.9),
                             'SVR_krPolc0.1e0.9':SVR(kernel="poly",C=0.1,epsilon=0.9),
                             'EN':ElasticNetCV(),
                             'ENcv3rs1':ElasticNetCV(cv=3, random_state=1),
                             'KN':KNeighborsRegressor(),
                             'KNn4aBallTree':KNeighborsRegressor(n_neighbors=4, algorithm="ball_tree"),
                             'KNn4aKdTree':KNeighborsRegressor(n_neighbors=4, algorithm="kd_tree"),
                             'KNn4aBrute':KNeighborsRegressor(n_neighbors=4, algorithm="brute"),
                             'GBoost': GradientBoostingRegressor(),
                             'GBoost_Dep5Est10': GradientBoostingRegressor(n_estimators=10, max_depth=5),
                             'LightGBM' : LGBMRegressor(),
                             'LightGBMRs42' : LGBMRegressor(random_state=42),
                             'XGBoost': XGBRegressor(),
                             'XGBoostObjSqerRs42': XGBRegressor(objective="reg:squarederror", random_state=42)}
    return list_regression_models



# training dataframe
inputTableTrainTest = 'vw_food_import_train_test'
connection = create_engine('postgresql://postgres:p0stgres@10.193.20.15:5432/adda')
cursor = connection.connect()
# query = "select * from "+inputTableTrainTest
query = "select * from "+inputTableTrainTest + " where item_country in ('CHICKEN MEAT - FROZEN~ARGENTINA', 'CHICKEN MEAT - FROZEN~BRAZIL') "
df_train = pd.read_sql_query(sql_text(query), cursor)
connection.dispose()


# Main training and prediction common hyper parameters
dateCol = 'imp_date'
dowCol = 'day'
areaFeature = 'item_country'
targetFeature = "import_kg"
categoricalFeatures = ['imp_month']
ignoreInputFeatures = [targetFeature, dowCol, dateCol, areaFeature, 'imp_month']
inputFeaturesForAutoRegression = []
targetWeekLag = 0
targetDayLag = 24
inputsWeekLag = 0
inputsDayLag = 6
train_ratio = 0.8
outlierSTD = 6   #high outlier filtering will be done with 6 std - if any outlier still detected, theyt will be replaced by median
#corr = 0

# create a dictionary of hyperparameter for saving into a file - to be reloaded for prediction
dict_hp = {'dateCol': dateCol, 'dowCol': dowCol, 'areaFeature': areaFeature, 'targetFeature': targetFeature,
               'categoricalFeatures': categoricalFeatures, 'ignoreInputFeatures': ignoreInputFeatures,
               'inputFeaturesForAutoRegression': inputFeaturesForAutoRegression, 'targetWeekLag': targetWeekLag,
               'targetDayLag': targetDayLag, 'inputsWeekLag': inputsWeekLag, 'inputsDayLag': inputsDayLag,
               'train_ratio': train_ratio, 'outlierSTD': outlierSTD}


# Process initial dates - train test data
df_train[dateCol] = pd.to_datetime(df_train[dateCol])
df_train[dowCol] = df_train[dateCol].dt.day_name()

# create regression models to be tested
list_regression_models = create_custom_regression_models()

# run train test
dict_ohmodels_catFtr_areas, dict_best_model_areas, df_train_areas, df_acc_areas = tpr.run_training_by_area(df_train, dict_hp, list_regression_models)

# print output
dict_df_train_area = dict(iter(df_train_areas.groupby(areaFeature)))
dict_df_acc_area = dict(iter(df_acc_areas.groupby(areaFeature)))
for key in dict_df_train_area:
    df_acc_area = dict_df_acc_area[key].sort_values(by=['wape'], ascending=True).copy()
    ohmodels_catFtr_area = dict_ohmodels_catFtr_areas[key]
    best_model_area = dict_best_model_areas[key]
    df_train_area = dict_df_train_area[key].copy()
    df_train_area = df_train_area[[dateCol, 'actual', 'predicted']]
    df_train_area.set_index(dateCol, inplace=True)

    # Print or save above to DB
    print(key + '\tBest model: \t' + df_acc_area['model'].iloc[0] + '\t' + str(df_acc_area['wape'].iloc[0]) + '\t' + str(df_acc_area['r2'].iloc[0]))
    df_train_area.plot(figsize=(15, 3), kind='line', title = key +' Best Model : ' + df_acc_area['model'].iloc[0] + ' = ' + str(df_acc_area['wape'].iloc[0]) + '(' + str(df_acc_area['r2'].iloc[0]) + ')' )

plt.show()


# store hyper_parameters, transformer and main model as pkl file
with open('iPredict_17/dict_hp.pkl', 'wb') as f:
    pickle.dump(dict_hp, f)

with open('iPredict_17/dict_ohmodels_catFtr_areas.pkl', 'wb') as f:
    pickle.dump(dict_ohmodels_catFtr_areas, f)

with open('iPredict_17/dict_best_model_areas.pkl', 'wb') as f:
    pickle.dump(dict_best_model_areas, f)



