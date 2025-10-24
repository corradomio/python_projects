
import pandas as pd
import numpy as np
import iPredict_17.autoreg.AutoRegHelper as ar

#******* PredictHelper - Start ****************

def runPredictionForNewPeriod_new(cur_period, df_in, model, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dowCol, targetFeature, inputFeaturesForAutoRegression, ignoreInputFeatures, printLog):
    # expand (in this case correct) autoregressive feature for any newly updated rows - but it currently does for all rows again 
    df = ar.expand_autoregressive_features(targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, df_in, dowCol, targetFeature, inputFeaturesForAutoRegression)
    #print(df)         

    #Take that current row out for prediction    
    df3 = df.iloc[[cur_period]]
    #print(df3)

    # [CM] There is an ERROR here: WHY df3 contains the 'targetFeature' column?
    #       It seems that the colum is not removed from training!!!
    # create training and testing sets
    df3_X = df3[df3.columns.difference(ignoreInputFeatures + [targetFeature])]
    assert targetFeature not in df3_X.columns
    result = model.predict(df3_X)

    # finally update the predicted value in the current period
    df.at[df.index[cur_period], targetFeature] = list(result)[0]
    return df

def exitIfNotEnoughDataInPredictionSource_new(df, targetFeature, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag):
    requiredDataLength = np.amax([targetWeekLag * 7, targetDayLag, inputsWeekLag * 7, inputsDayLag])
    suppliedDataLength = len(df[df[targetFeature].notnull()].index)
    if (suppliedDataLength < requiredDataLength): 
        import sys
        sys.exit("Required "+str(requiredDataLength)+" days for timeseries autoregression, but found " + str(suppliedDataLength) + " days in source data for prediction. Increase prediction data size or decrease autoregressive length. Exiting." )
        
def runPredictionForAllNewPeriods_new(df, model, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dowCol, targetFeature, inputFeaturesForAutoRegression, ignoreInputFeatures, printLog):
    suppliedData_Period = len(df[df[targetFeature].notnull()].index)
    total_Period = len(df.index)
    pred_period = total_Period - suppliedData_Period
    for i in range(pred_period):
        cur_period = suppliedData_Period + i
        df = runPredictionForNewPeriod_new(cur_period, df, model, targetWeekLag, targetDayLag, inputsWeekLag, inputsDayLag, dowCol, targetFeature, inputFeaturesForAutoRegression, ignoreInputFeatures, printLog)
    return df

#******** PredictHelper - End *****************
