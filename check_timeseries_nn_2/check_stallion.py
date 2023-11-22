import warnings
import logging.config

from matplotlib import pyplot as plt
from sktimex.utils import plot_series

import pandasx as pdx
import sktime.forecasting.naive
import sktimex


def main():
    df_all = pdx.read_data(
        f"./data/stallion.csv",
        datetime=['date', '%Y-%m-%d', 'M'],
        index=['agency', 'sku', 'date'],
        ignore=['timeseries', 'agency', 'sku', 'date'] + [
            'industry_volume', 'soda-volume',
            'avg_population_2017', 'avg_yearly_household_income_2017'
        ],
        binary=["easter_day",
                "good_friday",
                "new_year",
                "christmas",
                "labor_day",
                "independence_day",
                "revolution_day_memorial",
                "regional_games",
                "fifa_u_17_world_cup",
                "football_gold_cup",
                "beer_capital",
                "music_fest"
                ]
    )

    scaler = pdx.MinMaxScaler()
    df_all = scaler.fit_transform(df_all)

    df_dict = pdx.groups_split(df_all)
    keys = list(sorted(df_dict.keys()))
    for key in keys:
        df = df_dict[key]
        X, y = pdx.xy_split(df, target='volume')

        X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

        # f = sktimex.forecasting.linear.LinearForecaster(
        #     lags=(3, 3),
        #     tlags=3
        # )

        f = sktimex.forecasting.lnn.LinearNNForecaster(
            lags=(3, 3),
            tlags=3,
        )

        f.fit(y=y_train, X=X_train)

        # Xh, yh = None, None
        Xh = X_train
        yh = y_train
        sktimex.clear_yX(f)

        y_predict = f.predict_history(fh=y_test.index, X=X_test, yh=yh, Xh=Xh)

        plot_series(y_train['volume'], y_test['volume'], y_predict['volume'])
        plt.show()
        break


    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
