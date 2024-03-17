import logging.config
import pandasx as pdx
import sktimex as sktx
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series
from sktimex.forecasting import CNNLinearForecaster, RNNLinearForecaster, LinearForecaster

TARGET = 'import_kg'  # target column
GROUP = 'item_country'  # time series
YEARS = 7  # data collected years
DATETIME = 'imp_date'
MODEL = 'model'


class Data:

    def __init__(self):
        self.data = {}
        pass

    def load(self):
        df = pdx.read_data('vw_food_import_train_test_newfeatures.csv',
                           datetime=('imp_date', '[%Y/%m/%d %H:%M:%S]', 'M'),
                           onehot=['imp_month'],
                           ignore=['imp_month', 'prod_kg', 'avg_retail_price_src_country',
                                   'producer_price_tonne_src_country',
                                   'max_temperature', 'min_temperature'],
                           numeric=['evaporation', 'mean_temperature', 'rainy_days', 'vap_pressure'],
                           # periodic=('imp_date', 'M'),
                           na_values=['(null)'])

        dfdict = pdx.groups_split(df, groups=GROUP, drop=True)

        for key in dfdict:
            df = dfdict[key]

            df = pdx.set_index(df, index='imp_date', drop=True)
            df = pdx.clip_outliers(df, columns=TARGET)

            train, test = pdx.train_test_split(df, test_size=12)
            X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target=TARGET)
            fh = ForecastingHorizon(y_test.index, is_relative=False)

            self.data[key] = (X_train, y_train, X_test, y_test, fh)
        # end
    # end

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, item):
        return self.data.__getitem__(item)

# end

SELECTED = None


def main():

    data = Data()
    data.load()

    for key in data:
        item_country = key[0]
        if SELECTED is not None and not item_country in SELECTED:
            continue

        X_train, y_train, X_test, y_test, fh = data[key]

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
        model = CNNLinearForecaster(hidden_size=12, scale=True)

        model.fit(X=X_train, y=y_train)
        y_pred = model.predict(fh=fh, X=X_test)

        plot_series(y_train['import_kg'], y_test['import_kg'], y_pred['import_kg'], labels=['train', 'test', 'pred'])
        plt.show()

    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
