import warnings
from sklearn.linear_model import *

import pandasx as pdx
from sktime.forecasting.fbprophet import Prophet
from sktimex.forecasting.scikit import ScikitForecaster

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    df = pdx.read_data("vw_food_import_aed_train_test_mini.csv",
                       datetime=['imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'],
                       numeric=['import_aed'],
                       onehot=['imp_month'],
                       ignore=['item_country', 'imp_date', 'imp_month'],
                       index=['item_country', 'imp_date'])

    df = pdx.groups_select(df, values='ANIMAL FEED~ARGENTINA', drop=False)

    X, y = pdx.xy_split(df, target='import_aed')
    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

    #
    # c'e' un errore alla riga 296 di
    #   D:\Python\envs\ipredict\Lib\site-packages\prophet\forecaster.py
    #
    #   if name not in df:          name e' 'import_aed' che NON E' presente in df
    #       raise ValueError(...)
    #
    # f = Prophet()
    # f.fit(X_train, y_train)
    # y_pred = f.predict(fh=list(range(1, 13)), X=X_test)
    #

    for estimator in [
        # SGDRegressor,
        # HuberRegressor,
        # TheilSenRegressor,
        # PoissonRegressor,
        # TweedieRegressor,
        # GammaRegressor
        LinearRegression,
        Ridge,
        ElasticNet,
        Lars,
        Lasso,
        LassoLars,
        ARDRegression,
        BayesianRidge,
    ]:
        try:
            f = ScikitForecaster(
                estimator=estimator,
                window_length=12,
                strategy='recursive'
            )
            f.fit(y=y_train, X=X_train)
            y_pred = f.predict(fh=list(range(1, 13)), X=X_test)
            print(estimator, "ok")
        except Exception as e:
            print(estimator, e)


    pass


if __name__ == "__main__":
    main()
