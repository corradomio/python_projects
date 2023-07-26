import pandasx as pdx
from sktime.utils.plotting import plot_series
import matplotlib.pyplot as plt

from sktimex import NeuralNetForecaster


def main():
    df = pdx.read_data(
        "D:/Dropbox/Datasets/kaggle/airline-passengers.csv",
        datetime=('Month', '%Y-%m', 'M'),
        ignore=['Month'],
        index=['Month'])
    # print(len(df))

    y = df[["#Passengers"]]
    y_train = y.iloc[:-24]
    y_test = y.iloc[-24:]

    nnf = NeuralNetForecaster(
        flavour='RNN',
        periodic='M',
        scale=True,
        steps=20,
        lr=0.001,
        criterion="torch.nn.MSELoss",
        optimizer="torch.optim.Adam",
        hidden_size=20,
        batch_size=16,
        max_epochs=500,
        patience=20)

    nnf.fit(y=y_train, X=None)
    y_pred = nnf.predict(fh=len(y_test))

    plot_series(y_train, y_test, y_pred, labels=["train", "test", "pred"])
    plt.show()

    pass


if __name__ == "__main__":
    main()
