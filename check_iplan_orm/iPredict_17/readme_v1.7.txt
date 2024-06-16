version 1.1

1. oh encoding - fix - cover the case when there is two cardinality for catagorical column in lb.classes_ but levelBinarizer only transfomrs into single column 0/1 values, in this case create and only pass single column header, otherwise pass all lb.classes_ as coumn header. It was throwing error for such catagorical columns

2. correct bug in findBestModel - initilaised to max number
        bestWape = float('inf')

3. Added print option to runModelsGetResults(list_regression_models, train_X, train_Y, test_X, test_Y, printLog):


version 1.2

1. added runModelsGetResultsFS (with Feature selection step) - not efficient at present - feature selection step prefermod repeateadly - need to do only once

2. Added code to downcast to int8 (1 byte) for memory saving for all one hot incoded features in ohEncodeCatFeatureWithGivenModel() - by default it would be int32  - could 
be further downcasted to 1bit

3. Added function to read csv with  memory minimised - to be used with caution
        def read_csv_memory_opt(path)

4. index_col parameter added to methodd - by default this parameter will not be passed and will be taken as index_col = None if there is no index column in csv, if firt column is index as is the case with csv where the index is saved 
by mistake use index_col = [0]
        def read_csv_memory_opt(path, index_col=None):

5. max_features parameter added to featureSelectLasso(df_X, df_y, alpha, max_features)

6. New package added for plotting - PlotHelper. 

7. One function added to plotHelper, to do multiple plotting for secondary access
# multi plot dataframe - each column on its own axis - very useful
# taken from https://stackoverflow.com/questions/11640243/pandas-plot-multiple-y-axes 
def plot_multi(data, cols=None, spacing=.1, **kwargs):

vertion 1.3

1. Renamed Plotter package to io, and added one more heper file - ReadWriteHelper.py wiht one new function - 
        read_csv_memory_opt(path, index_col=None):

2. Added new function to traintest -  Given a dataframe and other input parameters - perform training using all selected models and return results for best model 
        runTrainTestAllModels(df, targetFeature, categoricalFeatures, ignoreInputFeatures, train_ratio, outlierSTD, labelFeatures, list_regression_models):

3. Function changed to take train_ratio rather than train_size 
        -> def createTrainTestSets(df, train_size, targetFeature, ignoreInputFeatures):
        -> def createTrainTestSets(df, train_ratio, targetFeature, ignoreInputFeatures):

version 1.4

1. TrainPredictioRunner.py added to run the training and pridiction, usefull for multiple area run with minimal coding

version 1.5

1. Outlier removal code changed to not remove the record if the target value is a outlier, but to replace the target value by median

version 1.6

1. Hyperparameter kept in a single dictionary - simplifed the code
2. Saperated train and prediction pipeline

Version 1.7

1. Added method to run train test for multiple area into the core ipredict code
1. Added method to run predict for multiple area into the core ipredict code