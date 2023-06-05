import logging.config
import pandasx as pdx
import sktimex as sktx
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series


# dtype=None,
# categorical=[],
# boolean=[],
# numeric=[],
# index=[],
# ignore=[],
# onehot=[],
# datetime=None,
# periodic=None,
# count=False,
# dropna=False,
# reindex=False,
# na_values=None,


def main():
    dfall = pdx.read_data("data/stallion_all.csv",
                       datetime=('date', '%Y-%m-%d', 'M'),
                       # index=['agency', 'sku', 'date'],
                       ignore=['timeseries',
                               'avg_population_2017', 'avg_yearly_household_income_2017'],
                       )

    dfdict = pdx.groups_split(dfall, groups=['agency', 'sku'], keep=1)
    df = pdx.groups_merge(dfdict, groups=['agency', 'sku'])
    df = pdx.set_index(df, index=['agency', 'sku', 'date'], drop=True)

    train, test = pdx.train_test_split(df, test_size=12)
    X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target='volume')
    fh = ForecastingHorizon(np.arange(1, 13), is_relative=True)

    # --

    model = sktx.SimpleRNNForecaster(periodic='M', max_epochs=300)
    model.fit(y=y_train, X=X_train)
    nn_pred = model.predict(fh=fh, X=X_test)
    y_pred = nn_pred

    # --

    # model = sktx.LinearForecastRegressor()
    # model.fit(y=y_train, X=X_train)
    # lin_pred = model.predict(fh=fh, X=X_test)
    # y_pred = lin_pred

    # --

    # model = sktx.ScikitForecastRegressor(window_length=1)
    # model.fit(y=y_train, X=X_train)
    # skt_pred = model.predict(fh=fh, X=X_test)
    # y_pred = skt_pred

    # --

    # model = sktx.ScikitForecastRegressor(class_name='sktime.forecasting.arima.AutoARIMA')
    # model.fit(y=y_train, X=X_train)
    # skt_pred = model.predict(fh=fh, X=X_test)
    # y_pred = skt_pred

    # --

    ydict = pdx.index_split(y_test)
    for key in ydict:
        ytr = y_train.loc[key]
        yts = y_test.loc[key]
        ypr = y_pred.loc[key]

        plot_series(ytr, yts, ypr, labels=['train', 'test', 'pred'])
        plt.show()
    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
