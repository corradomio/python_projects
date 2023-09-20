import logging.config
import os

import matplotlib.pyplot as plt
import torch.nn
from sklearn.preprocessing import MinMaxScaler
from sktime.utils.plotting import plot_series
from pandasx.encoders import MinMaxEncoder

import pandasx as pdx
import sktimex


def main():
    os.makedirs("./plots", exist_ok=True)

    mms = MinMaxScaler()

    data = pdx.read_data('Cucumber.csv',
                         categorical=["prod_month"],
                         datetime=('Date', "%m/%d/%Y", 'M'),
                         ignore=['Date'],
                         index="Date")

    scaler = MinMaxEncoder('Production', (0, 10))
    data = scaler.fit_transform(data)

    print(data)

    train, test = pdx.train_test_split(data, train_size=0.8)
    X_train, y_train, X_test, y_test = pdx.xy_split(train, test, target='Production')

    # model = sktimex.LagsRNNForecaster(
    #     lags=[3, 12],
    #     flavour='lstm',
    #     hidden_size=20,
    #     criterion=torch.nn.MSELoss,
    #     optimizer=torch.optim.Adam,
    #     batch_size=16,
    #     max_epochs=1000
    # )
    model = sktimex.SimpleRNNForecaster(
        lags=[12, 12],
        flavour='lstm',
        hidden_size=20,
        criterion=torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        batch_size=1,
        max_epochs=1000,
        # activation='elu'
        activation=True,
        lr=0.001
    )

    model.fit(y=y_train, X=X_train)
    predict = model.predict(fh=y_test.index, X=X_test)

    #y = pdx.groups_select(train, groups=["agency", "sku"], values=["Agency_01", "SKU_01"], drop=True)['volume']
    #y_test = pdx.groups_select(test, groups=["agency", "sku"], values=["Agency_01", "SKU_01"], drop=True)['volume']
    #y_pred = pdx.groups_select(predict, groups=["agency", "sku", "model"], values=["Agency_01", "SKU_01", "rnn-0-12"], drop=True)['volume']

    y = train['Production']
    y_test = test['Production']
    y_pred = predict['Production']

    plot_series(y, y_test, y_pred, labels=["y", "y_test", "y_pred"], title="Production")
    plt.show()
    # plt.savefig("./plots/Production-ipredict.png", dpi=300)


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
