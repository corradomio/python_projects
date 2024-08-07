import logging.config
import os

import matplotlib.pyplot as plt
from sktime.transformations.series.detrend import Detrender
from sktime.utils.plotting import plot_series

import pandasx as pdx
from sktimex import CNNLinearForecaster


def main():
    os.makedirs("./plots", exist_ok=True)

    data = pdx.read_data('airline.csv',
                         datetime=('Period', "%Y-%m", 'M'),
                         ignore='Period',
                         index="Period")
    y_orig = data

    det = Detrender()
    y = det.fit_transform(y_orig)

    plot_series(y_orig, y, labels=["y_orig", "y"], title="airline")
    plt.show()

    y_train, y_test = pdx.train_test_split(y, test_size=12)

    # -----------------------------------------------------------------------

    forecaster = CNNLinearForecaster(
        lags=[0, 12],
        flavour="cnn",

        hidden_size=10,

        lr=0.01,
        criterion="torch.nn.MSELoss",
        optimizer="torch.optim.Adam",
        activation=None,
        batch_size=8,
        max_epochs=1000,
        patience=100
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

