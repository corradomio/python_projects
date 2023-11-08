import warnings
import logging.config

from matplotlib import pyplot as plt
from sktime.utils import plot_series

import pandasx as pdx
import sktime.forecasting.naive


def main():
    df_all = pdx.read_data(
        f"./data/stallion.csv",
        datetime=['date', '%Y-%m-%d', 'M'],
        index=['agency', 'sku', 'date'],
        ignore=['timeseries', 'agency', 'sku', 'date'] + [
            'industry_volume', 'soda-volume'
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

    df_dict = pdx.groups_split(df_all)
    keys = list(sorted(df_dict.keys()))
    for key in keys:
        df = df_dict[key]
        X, y = pdx.xy_split(df, target='volume')

        X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

        f = sktime.forecasting.naive.NaiveForecaster(sp=3, strategy='last')
        f.fit(y=y_train, X=X_train)

        y_predict = f.predict(fh=y_test.index, X=X_test)

        plot_series(y_train['volume'], y_test['volume'], y_predict['volume'])
        plt.show()
        pass


    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
