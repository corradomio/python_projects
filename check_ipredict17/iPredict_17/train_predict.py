__all__ = [
    "train",
    "predict"
]

import logging
from typing import Any, Union, Optional
from typing import cast

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import MultiOutputMixin, RegressorMixin
from sklearn.linear_model._base import LinearModel

import iPredict_17.TrainingPredictionRunner as tpr


KEY_TYPE = Union[int, str]

# ---------------------------------------------------------------------------
# Interface to iPredict
# ---------------------------------------------------------------------------
# This file implements two functions:
#
#       train(df_train, hyper_params) -> dict_models
#
#  and
#
#       predict(df_predict, dict_models, hyper_params)
#
# to train the 'custom' models currently implemented in 'iPredict_<ver>'
#
# It is possible to specify a selected list of models using
#
#       train(df_train, hyper_params, models) -> dict_models
#
# where 'models' is a dictionary with the same structure of the file 'models_config.json'
#
# The goal of this module is to "normalize" how the data is passed in input
# and as result from 'iPredict_<ver>' module.
#
# There are several inconsistencies:
#
#   1)  it is necessary to pass a list of parameters instead then a simple
#       data structure or a simple dictionary containing the parameters
#
#   2) several parameters have very often always the same value.
#      Then, their values can be used as default
#
#   3)  there are problems when no model is able to generate a result.
#       Adding a "default models that works always" it is possible to
#       avoid this problem. Obviously, this model is very simple:
#       for now, it returns always zero!
#

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_HP = {
    # dataframe columns
    'areaFeature': None,                #
    'skillFeature': None,               #
    'dateCol': None,                    # timestamp
    'dowCol': None,                     # day of week (not necessary) OR name of the month

    # models parameters
    'targetFeature': None,              # to fill
    'categoricalFeatures': [],
    'ignoreInputFeatures': [],
    'inputFeaturesForAutoRegression': [],
    'targetWeekLag': 0,
    'targetDayLag': 7,
    'inputsWeekLag': 0,
    'inputsDayLag': 0,
    'train_ratio': .9,
    'outlierSTD': 6,
}

# Datetime column used in "IPlan" platform.
# It will be automatically replaced by <dateCol>, <dowCol>


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
        self._y_value: float = 0

    def get_params(self, deep=True):
        return super().get_params(deep)

    def fit(self, x=None, y=None, **kwargs):
        self._y_cols = 0 if len(y.shape) == 1 else y.shape[1]
        self._y_type = type(y)
        if isinstance(y, np.ndarray):
            self._y_name = None
        elif isinstance(y, pd.Series):
            self._y_name = y.name
        elif isinstance(y, pd.DataFrame):
            self._y_name = y.columns
        else:
            raise ValueError(f"Unsupported type {type(y)}")

        # self._y_value = y.median()
        # self._y_value = y.max()
        self._y_value = y.mean()
        return self

    def predict(self, x, **kwargs):
        if self._y_cols == 0:
            y = np.zeros(len(x))
        else:
            y = np.zeros((len(x), self._y_cols))

        y += self._y_value

        if self._y_name is None:
            return y
        if self._y_type == pd.Series:
            y = pd.Series(data=y, index=x.index, name=self._y_name)
        elif self._y_type == pd.DataFrame:
            y = pd.DataFrame(data=y, index=x.index, columns=self._y_name)
        else:
            raise ValueError(f"Unsupported type {type(x)}")
        return y


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def module_path():
    """
    Pyhon module of the current class
    """
    import sys
    import os.path
    this_path = sys.modules[__name__].__file__
    return os.path.dirname(this_path)


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
    kwargs = {} | config
    del kwargs['class']
    return klass(**kwargs)


def get_model_name(model_dict: dict) -> str:
    if 'best_model_name' in model_dict:
        return model_dict['best_model_name']
    else:
        return type(model_dict['best_model']).__name__


# ---------------------------------------------------------------------------
# Predefined list of models
# ---------------------------------------------------------------------------
# Note: to REPLACE using a configuration file!
#

def is_models_config(models_config: dict[str, Any]) -> bool:
    keys = list(models_config.keys())
    for k in keys:
        model = models_config[k]
        if type(model) in [str, dict]:
            return True
    return False
# end


def create_custom_regression_models() -> dict[str, Any]:
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
    #     # 'LightGBM': LGBMRegressor(),
    #     # 'LightGBMRs42': LGBMRegressor(random_state=42),
    #     'XGBoost': XGBRegressor(),
    #     'XGBoostObjSqerRs42': XGBRegressor(objective="reg:squarederror", random_state=42)
    # }

    models_config = json_load(f"{module_path()}/models_config.json")
    return create_regression_models(models_config)
# end


def create_regression_models(models_config) -> dict[str, Any]:
    list_regression_models = {}
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

def train(dft: pd.DataFrame, hp: Optional[dict]=None, models=None) \
    -> dict[
           tuple[KEY_TYPE, KEY_TYPE],
           dict[str, Any]
       ]:
    """

    To fill 'hp' at minimum with this configurations

        'targetFeature':    the target column

    :param DataFrame dft: dataframe used for the training.
        The dataframe must have the following columns:

            ['area_id_fk', 'skill_id_fk', 'state_date',  <inputFeatureId:str...>, <targetFeatureId:str>]
            ['area_id_fk', 'skill_id_fk', 'time', 'day', <inputFeatureId:str...>, <targetFeatureId:str>]

        where 'day' is the day of week as string ("Monday", ...)

    :param models: lit of models to use. If not specified, it is used a
        predefined list of models. It is a dictionary having the structures:

            {
                '<model_name>': <model_instance>,
                ...
            }


    :param hp: models hyper-parameters:

            areaFeature                         column containing the area
            skillFeature                        column containing the skill
            targetFeature                       column containing the target (univariate TS)

            dateCol                             column containing the date
            dowCol                              optional categorical column containing the day name

            categoricalFeatures                 list of categorical columns. It must contain
                                                dateCol, if not ignored
            ignoreInputFeatures                 list of columns to ignore
            inputFeaturesForAutoRegression      list ofcolumns to use for auto regression

            targetDayLag                        lags for target with step 1 (it can be used ALSO for monthly TS!)
            targetWeekLag                       lags for target with step 7
            inputsDayLag                        lags for input features with step 1, if there are input feature
            inputsWeekLag                       lags for input features with step 7, if there are input feature

            train_ratio                         train/test ratio used to evaluate the model
            outlierSTD                          multiplier for standard deviation used to remove outliers

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

        Note: 'df_train' and 'df_scores' contain the columns 'area_id_fk' and 'skill_id_fk'
        Note: 'df_train' contains the columns:

            'area_id_fk', 'skill_id_fk', 'state_date', 'actual', 'predicted'

    """
    log = logging.getLogger("ipredict.ipr17.train")
    log.info(f"Start training on df[{dft.shape}] ...")

    if hp is None: hp = {}

    # 1) compose the list of Hyper Parameters starting from the defaults and overriding
    #    with the parameters passed by argument
    dict_hp = DEFAULT_HP | hp

    # [DEBUG ONLY] check for null
    null_mask = dft.isnull().any(axis=1)
    null_rows = dft[null_mask]

    # 2) some consistency checks
    assert isinstance(dft, pd.DataFrame) and len(dft) > 0
    assert not dft[dft.columns.difference(dict_hp['ignoreInputFeatures'])].isnull().values.any(), \
        "The dataframe contains null values"
    assert isinstance(dict_hp['targetFeature'], (int, str)), "Missing mandatory 'targetFeature'"
    assert isinstance(dict_hp['areaFeature'], str), "Missing or wrong 'areaFeature' configuration"
    assert isinstance(dict_hp['skillFeature'], str), "Missing or wrong 'skillFeature' configuration"
    assert isinstance(dict_hp['dateCol'], str), "Missing or wrong 'dateCol' configuration"

    areaFeature = dict_hp['areaFeature']
    skillFeature = dict_hp['skillFeature']
    dateCol = dict_hp['dateCol']
    dowCol = dict_hp['dowCol']
    targetFeature = dict_hp['targetFeature']
    trainRatio = dict_hp['train_ratio']

    available_columns = dft.columns.intersection([areaFeature, skillFeature, dateCol, targetFeature])
    assert len(available_columns) == 4, \
           f"Missing some mandatory column: ('{areaFeature}', '{skillFeature}', '{dateCol}', '{targetFeature}') / {list(available_columns)}"

    # 2.1) add <dowCol> if defined but not present in the dataframe
    if dowCol is not None and dowCol not in dft.columns:
        dft[dowCol] = dft[dateCol].dt.day_name()

    # 3.1) add automatically 'dowCol' to the list of 'categoricalFeatures'
    if 'categoricalFeatures' not in dict_hp:
        dict_hp['categoricalFeatures'] = []
    if dowCol is not None and dowCol not in dict_hp['categoricalFeatures']:
        dict_hp['categoricalFeatures'].append(dowCol)

    # 3) extends 'ignoreInputFeatures' with 'areaFeature', 'skillFeature', 'targetFeature', 'dateCol'
    #    and all categorical features
    #    'dowCol' already added in the previous test
    dict_hp['ignoreInputFeatures'].extend([areaFeature, skillFeature, dateCol])

    # 5) split the dataframe based on the columns area/skill
    groups = [areaFeature, skillFeature]
    dict_of_regions_train = dict(iter(dft.groupby(groups)))

    # 6) prepare the dictionary containing ALL results
    #   (area, skill) -> { ... }
    models_dict: dict[tuple[int, int], dict[str, Any]] = {}

    # 6.1) prepare the list of 'df_train' used in different areas
    df_train_list = []

    for area_skill in dict_of_regions_train:
        log.info(f"... processing {area_skill}")

        area, skill = area_skill

        #
        # Retrieve the dataset to process and applies some data-preparation steps
        # Reset the index to avoid 'strange' behaviours in iPredict
        #
        df = dict_of_regions_train[area_skill]

        # Update 'train_ratio' parameter IF it is specified as a number > 1
        if trainRatio > 1:
            dict_hp['train_ratio'] = trainRatio/len(df)

        # Ensure the datime order
        # WARN: problems with an index with the same name than a column!
        # df.sort_values(by=dateCol, axis=1, inplace=True)

        # I don't understand this step!
        # df.fillna('NA', inplace=True)
        # Indeed!!!
        #   1) First all 'nan' are converted into 'NA'
        #   2) THEN into 0 (ZERO)
        #   3) THEN another time into 'NA'
        # BUT after the step 2) in the dataframe there will be NO 'nan' values
        #
        # [CM] at the begin of the function it is checked the dataframe 'null-free'
        # df.fillna(0, inplace=True)

        # [CM] I don't understand this step!
        #      Is it really possible that the training is done using FUTURE values???
        #      For now: COMMENTED!
        # df = df[~(df[dateCol] > pd.to_datetime('today'))]

        #
        # Create the list of models for this specific area/skill
        #
        if models is None:
            # create the models from the predefined list
            list_regression_models: dict[str, Any] = create_custom_regression_models()
        elif is_models_config(models):
            list_regression_models: dict[str, Any] = create_regression_models(models)
        else:
            # create the models based on 'models' parameter
            list_regression_models = {
                mname: sklearn.clone(models[mname])
                for mname in models
            }

        assert len(list_regression_models) > 0, "No models available for training"
        # log.info(f"... created {len(list_regression_models)} models: {list_regression_models.keys()}")

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
        log.info(f"... starting training ...")

        dict_ohmodels_catFtr_areas, dict_best_model_areas, df_train_areas, df_acc_areas \
            = tpr.run_training_by_area(df, dict_hp, list_regression_models)

        log.info(f"... training complete for {area_skill}")
        log.info(f"... collecting results")

        # This is IMPOSSIBLE for the presence of 'AllZerosModel'
        assert len(df_train_areas) > 0, "No predictions"

        # WARN: the following objects have a 'strange' structure:
        #
        #   dict_ohmodels_catFtr_areas: {<area>: <ohmodels_catFtr>}
        #        dict_best_model_areas: {<area>: <best_model>}
        #
        # for construction, the dictionaries will contain a SINGLE area
        # extract the data in more "intelligent" way

        ohmodels_catFtr = dict_ohmodels_catFtr_areas[area]
        best_model = dict_best_model_areas[area]

        # for 'consistency', the dataframe inside the dictionary MUST not contain information as,
        # for example, 'area_id_fk'
        # can be retrieved from the dictionary key.
        # The ALTERNATIVE is to ADD BOTH COLUMNS (area/skill) and NOT only one (area)!

        # For consistency, the 'skill_id_fk' is added to all dataframes
        df_train_areas[skillFeature] = skill
        df_acc_areas[skillFeature] = skill

        # sort the models scores in such way the best model is the first one
        # lower 'wape', bettermodel
        df_train = df_train_areas
        df_scores = df_acc_areas.sort_values(by=['wape'], ascending=True)

        area_skill: tuple[int, int] = cast(tuple[int, int], area_skill)
        models_dict[area_skill] = {
            "df_train": df_train,
            "df_scores": df_scores,
            "best_model_name": df_scores['model'].iloc[0],
            "best_wape": df_scores['wape'].iloc[0],
            "best_r_2": df_scores['r2'].iloc[0],
            "best_model": best_model,
            "ohmodels_catFtr": ohmodels_catFtr,
        }

        df_train_list.append(df_train)

        # [DEBUG ONLY]
        # break
    # end
    log.info(f"Done")

    return models_dict
# end


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def predict(dfp: pd.DataFrame, hp: dict, models: dict[tuple[int, int], dict[str, Any]]) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate the predictions for the dataframe 'dfp' using the models passed in 'models'

    :param dfp: dataframe used for the training.
        The dataframe must have the following columns:

            ['area_id_fk', 'skill_id_fk', 'state_date',  <inputFeatureId:str...>, <targetFeatureId:str>]
            ['area_id_fk', 'skill_id_fk', 'time', 'day', <inputFeatureId:str...>, <targetFeatureId:str>]

        where 'day' is the day of week as string

    :param models: models for area & skill. The dictionary must have the structure

        {
            (<area>, <skill>): {
                "best_model_name": <Model name>,
                "best_model": <ML Model trained>,
                "ohmodels_catFtr": <Python objects used as wrapper>,
            }
        }

        following exactly the same structure returned by 'train(...)'

    :param hp: hyperparameters used to train the models, and used for the prediction

    :return: a tuple composted by 2 datasets:

        tuple[0]: the dataframe passed in input with the NaN values replaced by the
                  the predicted values
        tuple[1]: the dataframe containing ONLY the predictions.

        The tuple[1] has the following columns:

            <areaFeature>, <skillFeature>, <dateCol>, 'actual', 'predicted'

        where 'actual' is always NaN

        Note: the dataframe returned is a clone of the one passed in input.
        Note: for now, the column <dowCol> is not included in the returnd dataframes
              because it seems to be not useful

    """
    log = logging.getLogger("ipredict.ipr17.predict")
    log.info(f"Start training on df[{dfp.shape}] ...")

    # 1) compose the list of Hyper Parameters starting from the defaults and overriding
    #    with the parameters passed by argument
    dict_hp = DEFAULT_HP | hp

    # 2) some consistency checks
    assert isinstance(dfp, pd.DataFrame) and len(dfp) > 0
    assert isinstance(dict_hp['targetFeature'], str), "Missing mandatory 'targetFeature'"
    assert isinstance(dict_hp['targetFeature'], str), "Missing or wrong 'targetFeature' configuration"
    assert isinstance(dict_hp['areaFeature'], str), "Missing or wrong 'areaFeature' configuration"
    assert isinstance(dict_hp['skillFeature'], str), "Missing or wrong 'skillFeature' configuration"
    assert isinstance(dict_hp['dateCol'], str), "Missing or wrong 'dateCol' configuration"

    areaFeature = dict_hp['areaFeature']
    skillFeature = dict_hp['skillFeature']
    dateCol = dict_hp['dateCol']
    dowCol = dict_hp['dowCol']
    targetFeature = dict_hp['targetFeature']

    assert len(dfp.columns.intersection([areaFeature, skillFeature, dateCol, targetFeature])) == 4, \
        f"Missing mandatory columns: '{areaFeature}', '{skillFeature}', '{dateCol}', '{targetFeature}'"

    # 2.1) add <dowCol> if defined but not present in the dataframe
    # WARN: this is WRONG. NOT all TS are 'daily'. For DEBUG ONLY
    if dowCol is not None and dowCol not in dfp.columns:
        dfp[dowCol] = dfp[dateCol].dt.day_name()

    # 3.1) extends 'ignoreInputFeatures' with 'areaFeature', 'skillFeature', 'dateCol', 'dowCol'
    dict_hp['ignoreInputFeatures'].extend([areaFeature, skillFeature, dateCol, dowCol])

    # 3.2) add automatically 'dowCol' to the list of 'categoricalFeatures'
    if dowCol is not None and dowCol not in dict_hp['categoricalFeatures']:
        dict_hp['categoricalFeatures'].append(dowCol)

    # 5) select the columns to use to split the dataframe based on area/skill
    groups = [areaFeature, skillFeature]
    dict_of_regions_train = dict(iter(dfp.groupby(groups)))

    # 6) prepare the dictionary containing ALL results
    Xy_list = []
    _y_list = []

    for area_skill in dict_of_regions_train:
        # check if area/skill is part of the training
        if area_skill not in models: continue

        log.info(f"... processing {area_skill}")

        area, skill = area_skill

        #
        # Retrieve the dataset to process and applies some data-preparation steps
        #
        df = dict_of_regions_train[area_skill]

        # ensure the datime order
        # WARN: problems with an index with the same name than a column
        # df.sort_values(by=dateCol, axis=1, inplace=True)

        # fill nan with zeros (as in training)
        # NO!!!! Otherwise it is not possible to find the locations of the data
        # to predict
        # df.fillna(0, inplace=True)

        #
        # Prediction
        #
        # {
        #     (<area>, <skill>): {
        #         "best_model_name": <Model name>,
        #         "best_model": <ML Model trained>,
        #         "ohmodels_catFtr": <Python objects used as wrapper>,
        #     }
        # }

        model_dict = models[cast(tuple[int, int], area_skill)]

        best_model_name = get_model_name(model_dict)
        dict_ohmodels_catFtr_areas = {area: model_dict["ohmodels_catFtr"]}
        dict_best_model_areas = {area: model_dict["best_model"]}

        log.info(f"... predictions for {area_skill} using {best_model_name}")

        # For 'defensive programming', it is mandatory to extract the indices
        # of the rows containing NaN in the target.
        # IN 'theory' 'y_pred' MUST have the same index!
        # nan_index = df[df[targetFeature].isnull()].index

        y_pred = tpr.run_prediction_by_area(df, dict_hp, dict_ohmodels_catFtr_areas, dict_best_model_areas)

        # WARN: the result DataFrame contains the following structure:
        #
        #       <areaFeature>, <dateCol>, <dowCol>, 'actual', 'predicted'
        #       ...            ...        ...       NaN       <value>
        #
        # It is necessary to FILL the dataframe passed in input with the predicted values
        log.info(f"... collecting results")

        # 1) clone 'df', because it could be reused
        #    In 'theory' the timestamps are 'consistent' between 'df_pred' and 'y_pred'
        #
        Xy_pred = df.copy()

        y_pred['predicted'] = y_pred['predicted'].astype(float)
        Xy_pred.loc[y_pred.index, targetFeature] = y_pred['predicted']

        # 2) for consistency 'y_pred' is reorganized to have a structure compatible with
        #    the dataframe passed in input
        #
        #       - 'actual' is removed (it contains NaNs)
        #       - 'predicted' is renamed <targetFeature>
        #
        # y_pred.drop(['actual', areaFeature], axis=1, inplace=True)
        # y_pred.rename(columns={'predicted': targetFeature}, inplace=True)

        # 3) prepare the output
        Xy_pred[areaFeature] = area
        Xy_pred[skillFeature] = skill
        Xy_list.append(Xy_pred)

        y_pred[areaFeature] = area
        y_pred[skillFeature] = skill
        _y_list.append(y_pred)
    # end

    #
    # Last step: concatenate ALL results
    #
    log.info(f"... finalizing predictions")

    if len(Xy_list) > 0:
        Xy_all = pd.concat(Xy_list)
        _y_all = pd.concat(_y_list)
    else:
        Xy_all = None
        _y_all = None

    #
    # Done!
    #
    log.info(f"Done")
    return Xy_all, _y_all
# end


# ---------------------------------------------------------------------------
# split_models_dict
# ---------------------------------------------------------------------------
#
# models_dict: dict[tuple, dict]
#   key: (area, skill)
#   value: dict[str, Any]
#       'best_model': ModelInstance()
#       'best_model_name': str
#       'best_r_2': float
#       'best_wape': float
#       'df_scores': DataFrame[columns=['country', 'item', 'model', 'wape', 'r2']]
#       'df_train' : dataFrame[columns=['country', 'item', 'date', 'actual', 'predicted']]
#       'ohmodels_catFtr': dict[str, Object]
#   end_dict
# end_dict


def split_models_dict(models_dict:dict) -> tuple[dict, dict]:
    """
    Split 'models_dict is a format compatible with the original
    implementation
    """
    dict_ohmodels_catFtr_areas = {}
    dict_best_model_areas = {}
    for k in models_dict:
        sk = "~".join(k)
        model_info = models_dict[k]
        dict_ohmodels_catFtr_areas[sk] = model_info['ohmodels_catFtr']
        dict_best_model_areas[sk] = model_info['best_model']
    # end

    return dict_ohmodels_catFtr_areas, dict_best_model_areas
# end


def merge_models_dict(dict_ohmodels_catFtr_areas, dict_best_model_areas) -> dict:
    models_dict = {}
    for sk in dict_ohmodels_catFtr_areas:
        k = tuple(sk.split('~')) if isinstance(sk, str) else sk

        best_model = dict_best_model_areas[sk]
        best_model_name = type(best_model).__name__

        models_dict[k] = {}
        models_dict[k]['ohmodels_catFtr'] = dict_ohmodels_catFtr_areas[sk]
        models_dict[k]['best_model'] = best_model
        models_dict[k]['best_model_name'] = best_model_name
    return models_dict
# end


# ---------------------------------------------------------------------------
# end
# ---------------------------------------------------------------------------


