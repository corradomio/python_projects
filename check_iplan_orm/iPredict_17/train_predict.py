# views.py
import logging
from typing import Any, cast
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel

import iPredict_17.TrainingPredictionRunner as tpr

# ---------------------------------------------------------------------------
# iPlan interface to iPredict
# ---------------------------------------------------------------------------
# This file implements two functions:
#
#       train(df_train, hyper_params)
#
#  and
#
#       predict(df_predict, dict_models, hyper_params)
#
# to train the 'custom' models currently implemented in 'iPredict_<ver>'
#

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_HP = {
    # dataframe columns
    'areaFeature': 'area_id_fk',        #
    'skillFeature': 'skill_id_fk',      # new
    'dateCol': 'time',                  # timestamp
    'dowCol': 'day',                    # day of week (not necessary)

    # models parameters
    'targetFeature': None,              # to fill
    'categoricalFeatures': ['day'],
    'ignoreInputFeatures': ['skill_id_fk', 'area_id_fk', 'time', 'day'],
    'inputFeaturesForAutoRegression': [],
    'targetWeekLag': 0,
    'targetDayLag': 7,
    'inputsWeekLag': 0,
    'inputsDayLag': 0,
    'train_ratio': .9,
    'outlierSTD': 6,

    # prediction parameters
    'startDate': None,                  # startDate
    'periodType': None,                 # period type: 'day', 'week', 'month'
    'periodLength': None,               # n of periods to predict
    'dataMaster': None                  # dataMaster
}

STATE_DATE = 'state_date'


# ---------------------------------------------------------------------------
# AllZerosModel
# ---------------------------------------------------------------------------
# This model never fails.
# In this way it is not necessary to check the number of predictions, because
# there will be at minimum one, THIS
#

class AllZerosModel(MultiOutputMixin, RegressorMixin, LinearModel):
    """
    Fake model returning all zeros.
    Used to be sure that at minimum a model it is able to return a prediction
    """
    def __init__(self):
        super().__init__()

        # == 0: y as single column
        #  > 0: y is a matrix with rows and columns
        self._y_cols: int = 0

    def fit(self, x=None, y=None, **kwargs):
        self._y_cols = 0 if len(y.shape) == 1 else y.shape[1]
        return self

    def predict(self, x, **kwargs):
        if self._y_cols == 0:
            y = np.zeros(len(x))
        else:
            y = np.zeros(len(x), self._y_cols)
        if isinstance(x, pd.Series):
            y = pd.Series(data=y, index=x.index)
        elif isinstance(x, pd.DataFrame):
            y = pd.DataFrame(data=y, index=x.index, columns=x.columns)
        return y


# ---------------------------------------------------------------------------
# Predefined list of models
# ---------------------------------------------------------------------------

def import_from(qname: str) -> Any:
    """
    Import a class specified by the fully qualified name string

    :param qname: fully qualified name of the class
    :return: Python class
    """
    import importlib
    p = qname.rfind('.')
    qmodule = qname[:p]
    name = qname[p+1:]

    module = importlib.import_module(qmodule)
    clazz = getattr(module, name)
    return clazz


def json_load(file: str, **kwargs) -> dict:
    import json
    with open(file, mode="r", **kwargs) as fp:
        return json.load(fp)


def create_model(name, config):
    klass = import_from(config['class'])
    params = {} | config
    del params['class']
    return klass(**params)


# ---------------------------------------------------------------------------
# Predefined list of models
# ---------------------------------------------------------------------------
# Note: to REPLACE using a configuration file!
#

def create_custom_regression_models():
    # from sklearn.neighbors import KNeighborsRegressor
    # from sklearn.linear_model import ElasticNetCV
    # from sklearn.svm import SVR
    # from sklearn.neural_network import MLPRegressor
    # from sklearn.linear_model import LinearRegression
    # from sklearn.tree import DecisionTreeRegressor
    # from sklearn import linear_model
    # from sklearn.ensemble import GradientBoostingRegressor
    # from lightgbm import LGBMRegressor
    # from xgboost import XGBRegressor
    #
    # list_regression_models = {
    #     'allzeros': AllZerosModel(),
    #     'BaysR': linear_model.BayesianRidge(),
    #     'Lasso': linear_model.Lasso(alpha=0.1),
    #     'DTR': DecisionTreeRegressor(random_state=0),
    #     'LinR': LinearRegression(),
    #     # 'LogR':LogisticRegression(),
    #     # 'LogR_c200pl2':LogisticRegression(C=100, penalty="l2"),
    #     'MLP': MLPRegressor(max_iter=1000),
    #     'MLP_hl33lr0.01es': MLPRegressor(hidden_layer_sizes=(3, 3),
    #                                      learning_rate_init=0.01,
    #                                      early_stopping=True,
    #                                      max_iter=1000),
    #     'SVR': SVR(),
    #     'SVR_krRBFc0.1e0.9': SVR(kernel="rbf", C=0.1, epsilon=0.9),
    #     # 'SVR_krLinc0.1e0.9':SVR(kernel="linear",C=0.1,epsilon=0.9),
    #     'SVR_krSigc0.1e0.9': SVR(kernel="sigmoid", C=0.1, epsilon=0.9),
    #     'SVR_krPolc0.1e0.9': SVR(kernel="poly", C=0.1, epsilon=0.9),
    #     'EN': ElasticNetCV(),
    #     'ENcv3rs1': ElasticNetCV(cv=3, random_state=1),
    #     'KN': KNeighborsRegressor(),
    #     'KNn4aBallTree': KNeighborsRegressor(n_neighbors=4, algorithm="ball_tree"),
    #     'KNn4aKdTree': KNeighborsRegressor(n_neighbors=4, algorithm="kd_tree"),
    #     'KNn4aBrute': KNeighborsRegressor(n_neighbors=4, algorithm="brute"),
    #     'GBoost': GradientBoostingRegressor(),
    #     'GBoost_Dep5Est10': GradientBoostingRegressor(n_estimators=10, max_depth=5),
    #     'LightGBM': LGBMRegressor(),
    #     'LightGBMRs42': LGBMRegressor(random_state=42),
    #     'XGBoost': XGBRegressor(),
    #     'XGBoostObjSqerRs42': XGBRegressor(objective="reg:squarederror", random_state=42)
    # }

    list_regression_models = {}
    models_config = json_load("models_config.json")
    for name in models_config:
        if name.startswith('#'):
            continue
        config = models_config[name]
        list_regression_models[name] = create_model(name, config)

    return list_regression_models
# end


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def train(dft: pd.DataFrame, hp: dict = {}) -> dict[tuple[int,int], dict[str, Any]]:
    """

    To fill 'hp' at minimum with this configurations

        'targetFeature':    the target column

    :param DataFrame dft: dataframe used for the training.
        The dataframe must have the following columns:

            ['area_id_fk', 'skill_id_fk', 'state_date',  <inputFeatureId:str...>, <targetFeatureId:str>]
            ['area_id_fk', 'skill_id_fk', 'time', 'day', <inputFeatureId:str...>, <targetFeatureId:str>]

        where 'day' is the day of week as string


    :return: a dictionary with structure:

        {
            (<area>, <skill>): {
                "best_model_name": <Model name>,
                "best_model": <ML Model trained>,
                "ohmodels_catFtr": <Python objects used as wrapper>,

                "df_train": <dataframe used for train>,
                "df_scores": <dataframe containing the algos' scores>,
            }
        }

    """
    log = logging.getLogger("ipredict.train")

    # 1) compose the list of Hyper Parameters starting from the defaults and overriding
    #    with the parameters passed by argument
    dict_hp = DEFAULT_HP | hp

    # 2) some consistency checks
    assert isinstance(dft, pd.DataFrame) and len(dft) > 0
    assert isinstance(dict_hp['targetFeature'], str), "Missing or wrong 'targetFeature' configuration"
    assert isinstance(dict_hp['areaFeature'], str), "Missing or wrong 'areaFeature' configuration"
    assert isinstance(dict_hp['skillFeature'], str), "Missing or wrong 'skillFeature' configuration"

    # 3) extends 'ignoreInputFeatures' with 'targetFeature'
    dict_hp['ignoreInputFeatures'].extend(dict_hp['targetFeature'])

    # 4) select the columns to use to split the dataframe based on area/skill
    groups = [dict_hp['areaFeature'], dict_hp['skillFeature']]
    dict_of_regions_train = dict(iter(dft.groupby(groups)))

    # 5) extract some values from the configuration
    areaFeature = dict_hp['areaFeature']
    dateCol = dict_hp['dateCol']
    dowCol = dict_hp['dowCol']

    # 6) prepare the dictionary containing ALL results
    #   (area, skill) -> { ... }
    results: dict[tuple[int, int], dict[str, Any]] = {}

    for area_skill in dict_of_regions_train:
        area, skill = area_skill

        #
        # Retrieve the dataset to process and applies some data-preparation steps
        #
        df = dict_of_regions_train[area_skill]

        # if 'df' contains 'state_date', replaces it with <dateCol> and add <dowCol>
        if STATE_DATE in df.columns:
            df.rename({STATE_DATE: dateCol}, inplace=True)
            df[dowCol] = df[dateCol].dt.day_name()

        # ensure the datime order
        df.sort_values(by=dateCol, inplace=True)

        # I don't understand this step!
        # df.fillna('NA', inplace=True)
        # Indeed!!!
        #   1) First all 'nan' are converted into 'NA'
        #   2) THEN into 0 (ZERO)
        #   3) THEN another time into 'NA'
        # BUT after the step 2) in the dataframe there will be NOT 'nan' values
        df.fillna(0, inplace=True)

        # I don't understand this step!
        # Is it really possible that the training is done using FUTURE values???
        df = df[~(df[dateCol] > pd.to_datetime('today'))]

        #
        # Create the list of models for this specific area/skill
        #
        list_regression_models = create_custom_regression_models()

        #
        # Train the models
        #
        # Input:
        #
        #   df: DataFrame[
        #       <skillFeature:str>, <areaFeature:str>, <dateCol:str>, <dowCol:str>,
        #       <inputFeaturesForAutoRegression:str...>,
        #       <targetFeature:str>
        #   ]
        #
        # Output:
        #
        #   dict_ohmodels_catFtr_areas: {
        #       '<area>': {
        #           '<dowCol>': <instance of TrainTestHelper.ohModel>
        #       }
        #   }
        #
        #   dict_best_model_areas: {
        #       '<area>': <instance of scikit-learn model>
        #   }
        #
        #   df_train_areas: DataFrame[<areaFeature:str>, <dateCol:str>, 'actual', 'predicted']
        #
        #   df_acc_areas: DataFrame[<areaFeature:str>, 'model', 'wape', 'r2']
        #
        #
        dict_ohmodels_catFtr_areas, dict_best_model_areas, df_train_areas, df_acc_areas = \
            tpr.run_training_by_area(df, dict_hp, list_regression_models)

        # This is IMPOSSIBlE for the presence of 'AllZerosModel'
        assert len(df_train_areas) > 0, "No predictions"

        # for construction, the dictionaries will contain a SINGLE area
        # extract the data in more "intelligent" way

        ohmodels_catFtr = dict_ohmodels_catFtr_areas[area]
        best_model = dict_best_model_areas[area]

        # for 'consistency', the dataframe inside the dictionary don't contain information
        # can be retrieved from the dictionary key.
        # The ALTERNATIVE is to ADD BOTH COLUMNS (area/skill) and NOT only one (area)!

        df_train = df_train_areas.drop(areaFeature, axis=1)
        df_scores = df_acc_areas.drop(areaFeature).sort_values(by=['wape'], ascending=True)

        area_skill: tuple[int, int] = cast(tuple[int, int], area_skill)
        results[area_skill] = {
            "df_train": df_train,
            "df_scores": df_scores,
            "best_model_name": df_scores['model'].iloc[0],
            "best_wape": df_scores['wape'].iloc[0],
            "best_r_2": df_scores['r2'].iloc[0],
            "best_model": best_model,
            "ohmodels_catFtr": ohmodels_catFtr,
        }

    return results
# end


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(dfp: pd.DataFrame, models: dict[tuple[int, int], dict[str, Any]], hp: dict = {}):
    """
    Generate the predictions for the dataframe 'dfp' using the models passed in 'models'

    :param dfp: dataframe used for the training.
        The dataframe must have the following columns:

            ['area_id_fk', 'skill_id_fk', 'state_date',  <inputFeatureId:str...>, <targetFeatureId:str>]
            ['area_id_fk', 'skill_id_fk', 'time', 'day', <inputFeatureId:str...>, <targetFeatureId:str>]

        where 'day' is the day of week as string

    :param models: models for area & skill. The dictionary mus have the structure

        {
            (<area>, <skill>): {
                "best_model_name": <Model name>,
                "best_model": <ML Model trained>,
                "ohmodels_catFtr": <Python objects used as wrapper>,
            }
        }

        following exactly the same structure returned by 'train(...)'

    :param hp: hyperparameters used to train the models, and used for the prediction

    """
    log = logging.getLogger("ipredict.predict")

    # 1) compose the list of Hyper Parameters starting from the defaults and overriding
    #    with the parameters passed by argument
    dict_hp = DEFAULT_HP | hp

    # 2) some consistency checks
    assert isinstance(dfp, pd.DataFrame) and len(dfp) > 0
    assert isinstance(dict_hp['targetFeature'], str), "Missing or wrong 'targetFeature' configuration"
    assert isinstance(dict_hp['areaFeature'], str), "Missing or wrong 'areaFeature' configuration"
    assert isinstance(dict_hp['skillFeature'], str), "Missing or wrong 'skillFeature' configuration"

    # assert isinstance(dict_hp['startDate'], (str, datetime)), "Missing or wrong 'skillFeature' configuration"
    # assert isinstance(dict_hp['periodType'], str), "Missing or wrong 'skillFeature' configuration"
    # assert isinstance(dict_hp['periodLength'], int), "Missing or wrong 'skillFeature' configuration"
    # assert isinstance(dict_hp['dataMaster'], (int, str)), "Missing or wrong 'dataMaster' configuration"

    # 3) extends 'ignoreInputFeatures' with 'targetFeature'
    dict_hp['ignoreInputFeatures'].extend(dict_hp['targetFeature'])

    # 4) select the columns to use to split the dataframe based on area/skill
    groups = [dict_hp['areaFeature'], dict_hp['skillFeature']]
    dict_of_regions_train = dict(iter(dfp.groupby(groups)))

    # 5) extract some values from the configuration
    areaFeature = dict_hp['areaFeature']
    dateCol = dict_hp['dateCol']
    dowCol = dict_hp['dowCol']
    dataMaster = dict_hp['dataMaster']

    # 6) prepare the dictionary containing ALL results
    results = {}

    for area_skill in dict_of_regions_train:
        area, skill = area_skill

        #
        # Retrieve the dataset to process and applies some data-preparation steps
        #
        df = dict_of_regions_train[area_skill]

        # if 'df' contains 'state_date', replaces it with <dateCol> and add <dowCol>
        if STATE_DATE in df.columns:
            df.rename({STATE_DATE: dateCol}, inplace=True)
            df[dowCol] = df[dateCol].dt.day_name()

        # ensure the datime order
        df.sort_values(by=dateCol, inplace=True)

        # fill nan with zeros (as in training)
        df.fillna(0, inplace=True)
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------
