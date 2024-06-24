import logging
import traceback

import iPredict_17.autoreg.AutoRegHelper as arh
import iPredict_17.traintest.TrainTestHelper as tth
import iPredict_17.autoreg.AutoRegPredictHelper as pdh
import pandas as pd
import numpy as np
import copy


TPR_LOGGER = None


def prepare_autoregressive_train_data(df, dict_hp):
    dowCol = dict_hp['dowCol']
    targetFeature = dict_hp['targetFeature']
    categoricalFeatures = dict_hp['categoricalFeatures']
    ignoreInputFeatures = dict_hp['ignoreInputFeatures']
    inputFeaturesForAutoRegression = dict_hp['inputFeaturesForAutoRegression']
    targetWeekLag = dict_hp['targetWeekLag']
    targetDayLag = dict_hp['targetDayLag']
    inputsWeekLag = dict_hp['inputsWeekLag']
    inputsDayLag = dict_hp['inputsDayLag']
    train_ratio = dict_hp['train_ratio']
    outlierSTD = dict_hp['outlierSTD']

    # Expand input features as required
    df = arh.expand_autoregressive_features(targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, df, dowCol, targetFeature, inputFeaturesForAutoRegression)

    # Remove all rows with null values arising from autoregressive feature expenssion
    df = arh.remove_null_values(targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, df, targetFeature, inputFeaturesForAutoRegression)

    #Filter Outliers    
    df = tth.outlier_filter(df,int(outlierSTD), targetFeature)

    #Perform Correlation analysis and only keep corelated features
    # not needed for now

    # build oh model for all specified categorical features    
    df, list_ohmodels_catFtr = tth.ohEncodeCatFeatureAll(df, categoricalFeatures)

    # create training and testing sets
    train, test, train_X, train_Y, test_X, test_Y = tth.createTrainTestSets(df, train_ratio, targetFeature, ignoreInputFeatures)
    
    return list_ohmodels_catFtr, train, test, train_X, train_Y, test_X, test_Y


def run_regression(train_X, train_Y, test_X, test_Y, list_regression_models, printLog):
    
    # Run each of the provided regresion algorithms
    list_regression_results = tth.runModelsGetResults(list_regression_models, train_X, train_Y, test_X, test_Y, printLog)

    #Find the best model and test result
    bestModelName, bestModel, bestWape, bestR2, bestDfPred = tth.findBestModel(list_regression_results)

    return list_regression_results, bestModelName, bestModel, bestWape, bestR2, bestDfPred


def prepare_autoregressive_pred_data(dfp, targetFeature, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dowCol, inputFeaturesForAutoRegression, list_ohmodels_catFtr):
    # find the prediction_period - number of null colums for target feature (total rows - all non null rows)
    suppliedData_Period = len(dfp[dfp[targetFeature].notnull()].index)
    total_Period = len(dfp.index)
    pred_period = total_Period - suppliedData_Period

    #check if the historical data included is enough for prediction if autoregressive days are set
    pdh.exitIfNotEnoughDataInPredictionSource_new(dfp, targetFeature, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag)

    # Expand input features as required
    dfp = arh.expand_autoregressive_features(targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dfp, dowCol, targetFeature, inputFeaturesForAutoRegression)

    #onehot encode all catagorical features 
    dfp = tth.ohEncodeCatFeatureWithGivenModelAll(dfp, list_ohmodels_catFtr)

    return pred_period, dfp


def run_training(df, dict_hp, list_regression_models):

    # prepare_autoregressive_train_data
    list_ohmodels_catFtr, train, test, train_X, train_Y, test_X, test_Y \
        = prepare_autoregressive_train_data(df, dict_hp)

    # Run training and find best model
    list_regression_results, bestModelName, bestModel, bestWape, bestR2, bestDfPred \
        = run_regression(train_X, train_Y, test_X, test_Y, list_regression_models, False)

    return test, bestDfPred, bestModelName, bestWape, bestR2, list_regression_results, list_ohmodels_catFtr, bestModel


def run_prediction(dict_hp, dfp, list_ohmodels_catFtr, bestModel):

    dowCol = dict_hp['dowCol']
    targetFeature = dict_hp['targetFeature']
    ignoreInputFeatures = dict_hp['ignoreInputFeatures']
    inputFeaturesForAutoRegression = dict_hp['inputFeaturesForAutoRegression']
    targetWeekLag = dict_hp['targetWeekLag']
    targetDayLag = dict_hp['targetDayLag']
    inputsWeekLag = dict_hp['inputsWeekLag']
    inputsDayLag = dict_hp['inputsDayLag']

    # prepare_autoregressive_pred_data
    pred_period, dfp = prepare_autoregressive_pred_data(dfp, targetFeature, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dowCol, inputFeaturesForAutoRegression, list_ohmodels_catFtr)

    # Run prediction
    dfp = pdh.runPredictionForAllNewPeriods_new(dfp, bestModel, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dowCol, targetFeature, inputFeaturesForAutoRegression, ignoreInputFeatures)

    return dfp, pred_period


def run_training_prediction(dict_hp, df, dfp, list_regression_models):

    # Prepare data and Run training
    test, bestDfPred, bestModelName, bestWape, bestR2, list_regression_results, list_ohmodels_catFtr, bestModel = run_training(df, dict_hp, list_regression_models)

    # Prepare data and Run prediction
    dfp, pred_period = run_prediction(dict_hp, dfp, list_ohmodels_catFtr, bestModel)

    return test, bestDfPred, bestModelName, bestWape, bestR2, list_regression_results, dfp, pred_period


def run_training_by_area(df_train, dict_hp, list_regression_models):
    dateCol = dict_hp['dateCol']
    areaFeature = dict_hp['areaFeature']
    targetFeature = dict_hp['targetFeature']
    targetDayLag = dict_hp['targetDayLag']

    # Init container to record final output for each areas
    df_acc_areas = pd.DataFrame()
    df_train_areas = pd.DataFrame()
    dict_ohmodels_catFtr_areas = {}
    dict_best_model_areas = {}

    # divide trainig data by each area into a dictionary
    dict_of_regions_train = dict(iter(df_train.groupby(areaFeature)))

    # Start loop for each area
    for key in dict_of_regions_train:
        curArea = key
        df = dict_of_regions_train[key].copy()

        # print("Running trainig and prediction for : " + str(curArea))

        #   check if all usable rows after lag removed have target equals to 0, if so skip that area
        #   This code could go into tpr.run_training_prediction a standard

        # [CM]
        # ALL ZEROS is a VALID list of values, and it is reasonable to predict ZERO!
        #
        # tmpdf = df.tail(len(df.index) - targetDayLag)
        # nonZeroValuesInUsableDataAfterLagRemoved = tmpdf[targetFeature].sum()
        # if nonZeroValuesInUsableDataAfterLagRemoved == 0:
        #     # print(str(curArea) + ' \t is skipped due to all usable target values being 0 ')
        #     continue

        # run main training and prediction for that area and get required outputs
        try:
            test, bestDfPred, bestModelName, bestWape, bestR2, list_regression_results, list_ohmodels_catFtr, bestModel \
                = run_training(df, dict_hp, list_regression_models)
        except Exception as e:
            exc = traceback.format_exc()
            # print(str(curArea) + ' \t is skipping, Something went wrong when trying to do training')
            continue

        # keep transformers and main model into a list - use deepcopy to copy trained model - otherwise it will point to final model
        dict_ohmodels_catFtr_areas[key] = list_ohmodels_catFtr
        dict_best_model_areas[key] = copy.deepcopy(bestModel)

        # prepare training output - accuracy
        df_acc = pd.DataFrame({areaFeature: curArea, 'model': key, 'wape': list_regression_results[key].wape,
                              'r2': list_regression_results[key].r2} for key in list_regression_results)
        # df_acc_areas = df_acc_areas.append(df_acc)
        df_acc_areas = pd.concat([df_acc_areas, df_acc], axis=0, join='outer')

        # prepare training output - actual vs prediction
        df_tr = df.join(bestDfPred)
        df_tr = df_tr[[areaFeature, dateCol, targetFeature, 'predicted']]
        df_tr = df_tr.rename(columns={targetFeature: 'actual'})
        # df_train_areas = df_train_areas.append(df_tr)
        df_train_areas = pd.concat([df_train_areas, df_tr], axis=0, join='outer')

    return dict_ohmodels_catFtr_areas, dict_best_model_areas, df_train_areas, df_acc_areas



def run_prediction_by_area(df_pred, dict_hp, dict_ohmodels_catFtr_areas, dict_best_model_areas):
    global TPR_LOGGER
    TPR_LOGGER = logging.getLogger("ipredict.sid")
    TPR_LOGGER.info("run_prediction_by_area")

    dateCol = dict_hp['dateCol']
    areaFeature = dict_hp['areaFeature']
    targetFeature = dict_hp['targetFeature']

    # divide trainig and prediction data by each area into a dictionary
    dict_of_regions_pred = dict(iter(df_pred.groupby(areaFeature)))

    # Init container to record final output for each areas
    df_pred_out = pd.DataFrame()

    # Start loop for each area
    for key in dict_of_regions_pred:
        curArea = key
        log = logging.getLogger(f"ipredict.sid.{curArea}")
        log.info("start processing")

        dfp = dict_of_regions_pred[key].copy()

        list_ohmodels_catFtr_reload = dict_ohmodels_catFtr_areas.get(curArea)
        if list_ohmodels_catFtr_reload is None:
            # print(str(curArea) + ' \t No categorical model found')
            log.warning(f"No categorical models found")
            continue

        bestModel_reload = dict_best_model_areas.get(curArea)

        # print('Best model pred:', key, ' - ', dict_best_model_areas.get(key))

        # print("Running prediction for : " + str(curArea))

        # run main prediction for that area and get required outputs
        try:
            log.info(f"run prediction using {bestModel_reload}")

            dfp, pred_period = run_prediction(dict_hp, dfp, list_ohmodels_catFtr_reload, bestModel_reload)
        except Exception  as e:
            # print(curArea + ' \t DATA_ERROR: cant perform prediction')
            log.error(f"prediction failed:")
            exc = traceback.format_exc()
            log.error(f"... {e}\n{exc}")
            continue

        # prepare prediction output
        # numpy version 1.8.0 is required for below to work
        pred = pd.DataFrame({areaFeature: curArea, dateCol: dfp[dateCol], 'predicted': dfp[targetFeature]})
        predToDB = pred.tail(pred_period)

        #   then create future prediction set, with nan for actuals, ready to be appended to train test set
        df_pred_area = predToDB[[areaFeature, dateCol, 'predicted']]
        df_pred_area["actual"] = np.nan
        df_pred_area = df_pred_area[[areaFeature, dateCol, 'actual', 'predicted']]
        df_pred_out = pd.concat([df_pred_out, df_pred_area], axis=0, join='outer')


    return df_pred_out
