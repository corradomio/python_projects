import logging.config
import os

import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series

import pandasx as pdx
from pandasx import MinMaxEncoder
import sklearn
from sktimex import ScikitForecastRegressor


def main():
    os.makedirs("./plots", exist_ok=True)

    data = pdx.read_data('airline.csv',
                         datetime=('Period', "%Y-%m", 'M'),
                         ignore='Period',
                         index="Period")
    y = data
    # mms = MinMaxEncoder('Number of airline passengers', feature_range=(0, 10))
    # y = mms.fit_transform(data)

    y_train, y_test = pdx.train_test_split(y, test_size=12)

    # -----------------------------------------------------------------------

    forecaster = ScikitForecastRegressor(
        lags=[0, 12],
        estimator=sklearn.linear_model.LinearRegression
    )

    print("train")
    forecaster.fit(y=y_train)

    print("predict")
    y_pred = forecaster.predict(fh=y_test.index)

    plot_series(y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"], title="airline")
    plt.show()
    # plt.savefig("./plots/stallion-rnn.png", dpi=300)

    pass

# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

