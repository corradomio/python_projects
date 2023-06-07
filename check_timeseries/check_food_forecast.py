import logging.config

import matplotlib.pyplot as plt
import pandasx as pdx
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktimex import SimpleCNNForecaster

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

TARGET = 'import_kg'  # target column
GROUP = 'item_country'  # time series
YEARS = 7  # data collected years
DATETIME = 'imp_date'
MODEL = 'model'


def prepare_data():
    df = pdx.read_data('vw_food_import_train_test_newfeatures.csv',
                       datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),
                       onehot=['imp_month'],
                       ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country',
                               'producer_price_tonne_src_country',
                               'max_temperature', 'min_temperature'],
                       numeric=['evaporation', 'mean_temperature', 'rainy_days', 'vap_pressure'],
                       # periodic=('imp_date', 'M'),
                       na_values=['(null)'])
    # df = pdx.dataframe_split_column(df, column=GROUP, columns=['item', 'country'], sep='~')
    # df = pdx.clip_outliers(df, columns=TARGET, groups=GROUP)

    dfdict = pdx.groups_split(df, groups=GROUP, drop=True)
    return dfdict


# SELECTED = ['ANIMAL FEED~MALAYSIA']
SELECTED = None


def main():
    dfdict = prepare_data()

    for key in dfdict:
        item_country = key[0]
        if SELECTED is not None and not item_country in SELECTED:
            continue

        df = dfdict[key]

        df = pdx.set_index(df, index='imp_date', drop=True)
        df = pdx.clip_outliers(df, columns=TARGET)

        train, test = pdx.train_test_split(df, test_size=12)
        X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target=TARGET)
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        # model = SimpleRNNForecaster(
        #     y_only=not features,
        #     flavour='lstm',
        #     periodic='M',
        #     scale=True,
        #     steps=12,
        #     lr=0.001,
        #     criterion="torch.nn.MSELoss",
        #     optimizer="torch.optim.Adam",
        #     hidden_size=20,
        #     batch_size=16,
        #     max_epochs=500,
        #     patience=20,
        # )
        # model = LinearForecastRegressor(lag=(1, 12))
        # model = ScikitForecastRegressor(window_length=1)
        # model = LinearForecastRegressor(lag=(0, 12))
        # model = LinearForecastRegressor(lag=(12, 12))
        # model = LinearForecastRegressor(lag=(0, 12), current=True)
        model = SimpleCNNForecaster(hidden_size=12, scale=True)

        model.fit(X=X_train, y=y_train)
        # t_train = model.predict(fh=y_train.index, X=X_train)
        y_pred = model.predict(fh=fh, X=X_test)

        plot_series(y_train['import_kg'], y_test['import_kg'], y_pred['import_kg'], labels=['train', 'test', 'pred'])
        plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
