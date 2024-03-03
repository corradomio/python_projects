import logging.config
import os

import matplotlib.pyplot as plt
import pandas as pd
from sktime.transformations.series.detrend import Detrender
from sktime.utils.plotting import plot_series

import pandasx as pdx
from pandasx.preprocessing import MinMaxNormalizer
from sktimex import RNNLinearForecaster


def main():
    os.makedirs("./plots", exist_ok=True)

    data = pdx.read_data('airline.csv',
                         datetime=('Period', "%Y-%m", 'M'),
                         ignore='Period',
                         index="Period")
    y_orig = data

    # det = Detrender(forecaster=PolynomialTrendForecaster(degree=1))
    det = Detrender()
    y = det.fit_transform(y_orig)

    plot_series(y_orig, y, labels=["y_orig", "y"], title="airline")
    plt.show()

    y_orig = y
    mmn = MinMaxNormalizer()
    y = mmn.fit_transform(y_orig)

    # plot_series(y_orig, y, labels=["y_orig", "y"], title="airline")
    # plt.show()

    plot_series(y, labels=["y"], title="airline")
    plt.show()

    y_train, y_test = pdx.train_test_split(y, test_size=12)

    # -----------------------------------------------------------------------

    forecaster = RNNLinearForecaster(
        lags=[0, 12],
        flavour="lstm",
        scale=True,

        num_layers=1,
        bidirectional=False,
        hidden_size=10,

        lr=0.01,
        criterion="torch.nn.MSELoss",
        optimizer="torch.optim.Adam",
        activation=None,
        batch_size=8,
        max_epochs=1000,
        patience=20
    )

    print("train")
    forecaster.fit(y=y_train)

    print("predict")
    y_pred = forecaster.predict(fh=y_test.index)

    fh_fore = pd.period_range(y_test.index[-1], periods=24)
    y_fore = forecaster.predict(fh=fh_fore)

    y_train = mmn.inverse_transform(y_train)
    y_test = mmn.inverse_transform(y_test)
    y_pred = mmn.inverse_transform(y_pred)
    y_fore = mmn.inverse_transform(y_fore)

    y_train = det.inverse_transform(y_train)
    y_test = det.inverse_transform(y_test)
    y_pred = det.inverse_transform(y_pred)
    y_fore = det.inverse_transform(y_fore)

    plot_series(y_train, y_test, y_pred, y_fore, labels=["y_train", "y_test", "y_pred", "y_fore"], title="airline")
    plt.show()
    # plt.savefig("./plots/stallion-rnn.png", dpi=300)

    pass

# end


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

