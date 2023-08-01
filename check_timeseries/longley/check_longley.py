import logging.config
import pandas as pd
import pandasx as pdx
import matplotlib.pyplot as plt
from sktime.datasets import load_longley
from sktime.forecasting.var import VAR
from sktime.utils.plotting import plot_series
from sklearn.linear_model import LinearRegression


def test1():
    # columns:
    #   y: TOTEMP
    #   X: GNPDEFL GNP UNEMP ARMED POP
    y, X = load_longley()

    # targets: TOTEMP GNPDEFL GNP
    df = pd.concat([X, y], axis=1)

    X, y = pdx.xy_split(df, target=['TOTEMP', 'GNPDEFL', 'GNP'])

    forecaster = VAR()
    forecaster.fit(y)

    y_pred = forecaster.predict(fh=[1, 2, 3])

    plot_series(y['TOTEMP'], y_pred['TOTEMP'], labels=['y', 'y_pred'], title='TOTEMP')
    plot_series(y['GNPDEFL'], y_pred['GNPDEFL'], labels=['y', 'y_pred'], title='GNPDEFL')
    plot_series(y['GNP'], y_pred['GNP'], labels=['y', 'y_pred'], title='GNP')
    plt.show()


def test2():


    pass


def main():
    # test1()
    test2()


    pass


if __name__ == "__main__":
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()

