
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#****** TrainTestHelper - Start ***************


    
def ohEncodeCatFeature(df, catFeature):
        lb = LabelBinarizer()
        oh_model = lb.fit(df[catFeature])   
        df = ohEncodeCatFeatureWithGivenModel(df, catFeature, oh_model, lb)        
        return df, oh_model, lb

def ohEncodeCatFeatureWithGivenModel(df, catFeature, oh_model, lb):
        transformedData = oh_model.transform(df[catFeature])
        
        # cover the case when there is two cardinality for catagorical column in lb.classes_ but levelBinarizer only transfomrs into single column 0/1 values
        # in this case create and only pass single column header, otherwise pass all lb.classes_ as coumn header
        numOhCols = len(transformedData[0])
        numOhCardinality = len(lb.classes_)
        if (numOhCols==1 and numOhCardinality==2):
            header = str(lb.classes_[0])+'-'+str(lb.classes_[1])
            dfc = pd.DataFrame(transformedData,columns=[header],index=df.index)
        else:
            dfc = pd.DataFrame(transformedData,columns=lb.classes_,index=df.index)

        dfc = dfc.add_prefix(catFeature + '_')
        
        # downcast to int8 (1 byte) for memory saving - could be further downcasted to 1bit
        for fname in dfc.columns:
            #dfc[fname] = pd.to_numeric(dfc[fname], downcast='unsigned')
            df[fname] = pd.to_numeric(dfc[fname], downcast='unsigned')
        
        #df = df.join(dfc)
        return df

class ohModel:
    def __init__(self, model_oh, lb):
            self.model_oh =  model_oh
            self.lb = lb
            
def ohEncodeCatFeatureAll(df, categoricalFeatures):
    list_ohmodels_catFtr = {}    
    for catFtr in categoricalFeatures:
        df, ohmodel_cat, lb_cat = ohEncodeCatFeature(df, catFtr)
        list_ohmodels_catFtr[catFtr] = ohModel(ohmodel_cat, lb_cat)
    return df, list_ohmodels_catFtr

def ohEncodeCatFeatureWithGivenModelAll(df, list_ohmodels_catFtr):
    for key in list_ohmodels_catFtr:
        ohmodel = list_ohmodels_catFtr[key].model_oh
        ohlb = list_ohmodels_catFtr[key].lb
        df = ohEncodeCatFeatureWithGivenModel(df, key, ohmodel, ohlb)
    return df


#Outlier Filter
#{ Mean +- (value) * standard deviation } is considered for outlier filter
def outlier_filter(df, x, targetFeature):
        mean = np.mean(df[targetFeature], axis=0)
        sd = np.std(df[targetFeature], axis=0)
        value = mean + (x * sd)
        #print(value)
        value_x = mean - (x * sd)
        #print(value_x)
        df["outlier"] = df[targetFeature].apply(
            lambda x: x <= value_x or x >= value
        )

        # df = df[df['outlier'] != True]
        # replace outliers with median - new in iPredict_v1.5
        median = np.median(df[targetFeature], axis=0)
        df.loc[df.outlier == True, targetFeature] = median

        df = df.drop(columns=['outlier'])
        return df


#Error Calculation
#Weighted Absolute Percentage Error is being considered for calculation of error
# ********** This is not WAPE it is MAPE ********

def weighted_absolute_percentage_error(actuals, predictions):
        totalActual = 0.0
        totalABSdiff = 0.0
        count = 0
        for value in actuals:
            totalActual = totalActual + value
            prediction = 0.0
            if count == len(predictions):
                break
            prediction = predictions[count]
            totalABSdiff = totalABSdiff + np.abs(value - prediction)
            count = count + 1
        if totalActual > 0:
            return totalABSdiff/totalActual
        return totalABSdiff

def createTrainTestSets(df, train_ratio, targetFeature, ignoreInputFeatures):
    train_size = int(len(df) * train_ratio)
    train = df[0:train_size]
    test = df[train_size:]
    train_X = train[train.columns.difference(ignoreInputFeatures)]
    train_Y = train[targetFeature]
    test_X = test[test.columns.difference(ignoreInputFeatures)]
    test_Y = test[targetFeature]
    return train, test, train_X, train_Y, test_X, test_Y

def trainTest(model, train_X, train_Y, test_X, test_Y):
        model.fit(train_X,train_Y)        
        return test(model, test_X, test_Y)

def test(model, test_X, test_Y):
        result = model.predict(test_X)
        wape = weighted_absolute_percentage_error(test_Y, list(result))
        r2 = r2_score(test_Y, list(result))
        df_pred = pd.DataFrame({'actual': test_Y, 'predicted':list(result)})
        return model, wape, r2, df_pred

class regression_result:
    def __init__(self, modelname, model, wape, r2, df_pred):
            self.modelname = modelname
            self.model = model
            self.wape = wape
            self.r2 = r2
            self.df_pred = df_pred
            
def runModelsGetResults(list_regression_models, train_X, train_Y, test_X, test_Y, printLog):
        list_regression_results = {}   
        cnt =0   
        for key in list_regression_models:
            cnt = cnt+1
            if printLog:
                print('\n'+str(cnt)+' Running -> ' + key + ':\n'+ str(list_regression_models[key]))
            model, wape, r2, df_pred = trainTest(list_regression_models[key], train_X, train_Y, test_X, test_Y)    
            list_regression_results[key] = regression_result(key, model, wape, r2, df_pred)
            if printLog:
                print(key+'\t\t'+str(list_regression_results[key].wape) + '\t' + str(list_regression_results[key].r2))
        return list_regression_results

def featureSelectLasso(train_X, train_Y, alpha, max_features):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso
    ls = Lasso(alpha=alpha).fit(train_X, train_Y)
    return SelectFromModel(ls, prefit=True, max_features = max_features)

def featureSelectLR(train_X, train_Y, max_features):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LinearRegression
    ls = LinearRegression().fit(train_X, train_Y)
    return SelectFromModel(ls, prefit=True, max_features = max_features)

def featureSelectRF(train_X, train_Y, max_features):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor 
    ls = RandomForestRegressor(n_estimators = 100).fit(train_X, train_Y)
    return SelectFromModel(ls, prefit=True, max_features = max_features)

def featureSelectTransform(modelFS, org_X):
    selectedFeatures = modelFS.get_support(indices=True)
    new_X = org_X.iloc[:,selectedFeatures]
    return new_X

def runModelsGetResultsFS(list_regression_models, train_X, train_Y, test_X, test_Y, alpha, printLog):
    modelFS = featureSelectLasso(train_X, train_Y, alpha, None)
    selectedFeatures = modelFS.get_support(indices=True)
    train_X_new = train_X.iloc[:,selectedFeatures]
    test_X_new = test_X.iloc[:,selectedFeatures]
    #train_X_new = modelFS.transform(train_X)   
    #test_X_new = modelFS.transform(test_X) 
    #print(test_X_new)  
    return runModelsGetResults(list_regression_models, train_X_new, train_Y, test_X_new, test_Y, printLog)

def findBestModel(list_regression_results):
        bestWape = float('inf')
        for key in list_regression_results:
            if bestWape > list_regression_results[key].wape:
                bestWape = list_regression_results[key].wape
                bestR2 = list_regression_results[key].r2
                bestModel = list_regression_results[key].model
                bestDfPred = list_regression_results[key].df_pred             
                bestModelName = key
        return bestModelName, bestModel, bestWape, bestR2, bestDfPred

def create_regression_models():
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.linear_model import ElasticNetCV
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn import linear_model
    from sklearn.ensemble import GradientBoostingRegressor
    list_regression_models ={'BaysR':linear_model.BayesianRidge(), 
                             'Lasso':linear_model.Lasso(alpha=0.1),
                             'DTR':DecisionTreeRegressor(random_state=0),
                             'LinR':LinearRegression(),
                             #'LogR':LogisticRegression(),
                             #'LogR_c200pl2':LogisticRegression(C=100, penalty="l2"),
                             'MLP':MLPRegressor(),
                             'MLP_hl33lr0.01es':MLPRegressor(hidden_layer_sizes=(3, 3), learning_rate_init=0.01,early_stopping=True),
                             'SVR':SVR(),
                             'SVR_krRBFc0.1e0.9':SVR(kernel="rbf",C=0.1,epsilon=0.9),
                             #'SVR_krLinc0.1e0.9':SVR(kernel="linear",C=0.1,epsilon=0.9),
                             'SVR_krSigc0.1e0.9':SVR(kernel="sigmoid",C=0.1,epsilon=0.9),
                             'SVR_krPolc0.1e0.9':SVR(kernel="poly",C=0.1,epsilon=0.9),
                             'EN':ElasticNetCV(),
                             'ENcv3rs1':ElasticNetCV(cv=3, random_state=1),
                             'KN':KNeighborsRegressor(),
                             'KNn4aBallTree':KNeighborsRegressor(n_neighbors=4,algorithm="ball_tree"),
                             'KNn4aKdTree':KNeighborsRegressor(n_neighbors=4,algorithm="kd_tree"),
                             'KNn4aBrute':KNeighborsRegressor(n_neighbors=4,algorithm="brute"), 
                             'GBoost_Dep5Est500': GradientBoostingRegressor(n_estimators=500, max_depth=5)  
                             }
    return list_regression_models


# Given a dataframe and other input parameters - perform training using all selected models and return results for best model 
def runTrainTestAllModels(df, targetFeature, categoricalFeatures, ignoreInputFeatures, train_ratio, outlierSTD, labelFeatures, list_regression_models):
    train_size = int(len(df) * train_ratio)
    
    #Filter Outliers    
    #df = tth.outlier_filter(df,int(outlierSTD), targetFeature)

    # build oh model for all specified categorical features    
    df, list_ohmodels_catFtr = ohEncodeCatFeatureAll(df, categoricalFeatures)

    # create training and testing sets
    train, test, train_X, train_Y, test_X, test_Y = createTrainTestSets(df, train_size, targetFeature, ignoreInputFeatures)
    
    # Run all models 
    list_regression_results = runModelsGetResults(list_regression_models, train_X, train_Y, test_X, test_Y, False)

    #Find the best model and test result
    bestModelName, bestModel, bestWape, bestR2, bestDfPred = findBestModel(list_regression_results)
    
    #Add lables to test results - like Date in this case to group by during visualisation
    bestDfPred[labelFeatures] = test[labelFeatures]
    
    #return best result
    return bestModelName, bestModel, bestWape, bestR2, bestDfPred, list_ohmodels_catFtr



#https://stackoverflow.com/questions/42658379/variance-inflation-factor-in-python
def sklearn_vif(exogs, data):
    '''
    This function calculates variance inflation function in sklearn way. 
     It is a comparatively faster process.

    '''
    # initialize dictionaries
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1/(1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif

def iterative_vif(df):
    df_vif= sklearn_vif(exogs=df.columns, data=df).sort_values(by='VIF',ascending=False)
    while (df_vif.VIF>5).any() ==True:
        red_df_vif= df_vif.drop(df_vif.index[0])
        df= df[red_df_vif.index]
        df_vif=sklearn_vif(exogs=df.columns,data=df).sort_values(by='VIF',ascending=False)
    return df 

    
    
#******* TrainTestHelper - End ****************
