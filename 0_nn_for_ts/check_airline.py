import matplotlib.pyplot as plt
from sktimex.utils.plotting import plot_series

import pandasx as pdx

TARGET = 'Number of airline passengers'


def detrend(df):
    for method in ['identity', 'linear', 'power', 'piecewise', 'stepwise']:
        t = pdx.DetrendTransformer(method=method, method_select='median')
        y_det = t.fit_transform(df)

        plot_series(y_det[TARGET], labels=[TARGET], title=method)
        plt.show()


def minmax(df):
    # for method in ['poly1', 'linear', 'piecewise', 'stepwise', 'global', 1.25]:
    for method in ['linear']:
        t = pdx.ConfigurableMinMaxScaler(method=method)
        y_det = t.fit_transform(df)

        plot_series(y_det[TARGET], labels=[TARGET], title=method)
        plt.show()


def main():
    df = pdx.read_data(
        "./data/airline.csv",
        datetime=('Period', '%Y-%m', 'M'),
        ignore=['Period'],
        index=['Period']
    )

    plot_series(df[TARGET], labels=[TARGET])
    plt.show()

    # detrend(df)
    minmax(df)

    pass


if __name__ == "__main__":
    main()
