import logging.config
import os

import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series

import pandasx as pdx
from jsonx import *
from sktimex import SimpleRNNForecaster, SimpleCNNForecaster


def main():
    os.makedirs("./plots", exist_ok=True)

    print("Load the data")
    data = pdx.read_data('stallion_all.csv',
                         datetime=('date', "%Y-%m-%d", 'M'),
                         ignore=["timeseries", "avg_population_2017", "avg_yearly_household_income_2017", "date"],
                         categorical=[
                             "easter_day",
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
                         ],
                         binary="auto",
                         index="date")

    # split the dataset in separated time series
    print('Split the data in separated time series based on the columns ["agency", "sku"]')
    dict_data = pdx.groups_split(data, groups=["agency", "sku"], drop=True)
    keys = list(dict_data.keys())
    print('List of available time series')
    print(list(keys))

    # select the first dataset
    print('Select the first time series')
    data = dict_data[keys[0]]

    print('Split the time series in train and test, whe test composed by last 12 months')
    train, test = pdx.train_test_split(data, test_size=12)

    print('Split the time series in X *features) and y (target) with target column "volume"')
    X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target='volume')

    # -----------------------------------------------------------------------

    print("Create a simple RNN based on LSTM")
    forecaster = SimpleRNNForecaster(
        lags=[1, 1],
        y_only=false,
        flavour="lstm",
        periodic="M",
        scale=true,
        steps=12,
        lr=0.001,
        criterion="torch.nn.MSELoss",
        optimizer="torch.optim.Adam",
        hidden_size=20,
        batch_size=16,
        max_epochs=500,
        patience=20
    )

    print("Train the RNN")
    forecaster.fit(y=y_train, X=X_train)

    print("Predict using the RNN")
    pred = forecaster.predict(fh=y_test.index, X=X_test)

    print("Generate the plot and save it in ./plots/stallion-rnn.png")
    y = train['volume']
    y_test = test['volume']
    y_pred = pred['volume']

    plot_series(y, y_test, y_pred, labels=["y", "y_test", "y_pred"], title="stallion")
    # plt.show()
    plt.savefig("./plots/stallion-rnn.png", dpi=300)

    # -----------------------------------------------------------------------

    print("Create a simple CNN based on LSTM")
    forecaster = SimpleCNNForecaster(
        lags=[1, 1],
        y_only=false,
        flavour="lstm",
        periodic="M",
        scale=true,
        steps=12,
        lr=0.001,
        criterion="torch.nn.MSELoss",
        optimizer="torch.optim.Adam",
        hidden_size=20,
        batch_size=16,
        max_epochs=500,
        patience=20
    )

    print("Train the CNN")
    forecaster.fit(y=y_train, X=X_train)

    print("Predict using the CNN")
    pred = forecaster.predict(fh=y_test.index, X=X_test)

    print("Generate the plot and save it in ./plots/stallion-cnn.png")
    y = train['volume']
    y_test = test['volume']
    y_pred = pred['volume']

    plot_series(y, y_test, y_pred, labels=["y", "y_test", "y_pred"], title="stallion")
    # plt.show()
    plt.savefig("./plots/stallion-cnn.png", dpi=300)

    pass
# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

