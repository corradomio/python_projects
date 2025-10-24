2024/07/04

1) 'print' replaced by a 'logger'
   the logger uses the namespaces

        ipredict.train
        ipredict.prediction
        ipredict.train_prediction

   If the library is used inside a more complex application, it is
   necessary to be able to activate/deactivate the logging.
   Using a logging system, it is possible to select the log's output
   (console, file, database, SNMP, ...)

2) added some logs to know the process evolution

3) when an exception occurs, it is logged the callstack

4) when  train/test/predict datasets are created, it is removed

        ignoreInputFeatures PLUS [targetFeature]

    - AutoRegPredictHelper.py, line 18
    - TrainTestHelper.py, lines 107, 109

5) added the modulo 'train_predict.py' containing 2 functions:

      train(df_train, hyper_parameters, models=None -> models_dict
    predict(df_pred,  hyper_parameters, models_dict) -> df_pred_filled, df_predictions

6)