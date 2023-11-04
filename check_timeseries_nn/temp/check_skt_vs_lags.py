import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sktime.datasets import load_airline
from sktime.forecasting.base import ForecastingHorizon
from sktime.utils.plotting import plot_series

import pandasx as pdx

# hide warnings
warnings.filterwarnings("ignore")


PLOTS_DIR = "./plots"
DATA_DIR = "../data"

DATETIME = ['date', "%Y-%m-%d", "M"]
TARGET = 'volume'
GROUPS = ['agency', 'sku']
TARGET_2 = 'volume2'


def plot_pred(y, y_pred):
    plot_series(y, y_pred, labels=["y", "y_pred"])
    plt.show()


def main():
    y = load_airline()

    # plotting for visualization
    plot_series(y)

    # fh = np.arange(1, 37)
    fh = ForecastingHorizon(
        pd.PeriodIndex(pd.date_range("1961-01", periods=36, freq="M")), is_relative=False
    )
    cutoff = pd.Period("1960-12", freq="M")

    fhr = fh.to_relative(cutoff)
    fha = fhr.to_absolute(cutoff)

    # -----------------------------------------------------------------------

    from sktime.forecasting.naive import NaiveForecaster
    forecaster = NaiveForecaster(strategy="last", sp=12)
    forecaster.fit(y)

    y_pred = forecaster.predict(fh)
    plot_pred(y, y_pred)

    # -----------------------------------------------------------------------

    from sktimex.forecasting import ScikitForecaster
    forecaster = ScikitForecaster(estimator=NaiveForecaster, strategy="last", sp=12)
    forecaster.fit(y)

    y_pred = forecaster.predict(fh)
    plot_pred(y, y_pred)

    # -----------------------------------------------------------------------

    from sktimex.forecasting import ScikitForecaster
    forecaster = ScikitForecaster(estimator="sktime.forecasting.naive.NaiveForecaster", strategy="last", sp=12)
    forecaster.fit(y)

    y_pred = forecaster.predict(fh)
    plot_pred(y, y_pred)

    pass


def main1():
    df = pdx.read_data(f"{DATA_DIR}/stallion.csv",
                       datetime=DATETIME,
                       index=GROUPS + DATETIME[0:1],
                       ignore=["timeseries"] + DATETIME[0:1] + GROUPS + [
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

    # df_sel1 = df.loc[('Agency_01',)]
    # df_sel2 = df.xs('Agency_01', level=0, drop_level=True)
    # df_sel3 = df.xs('SKU_01', level=1, drop_level=True)
    # df_sel4 = df.loc[('Agency_01', 'SKU_01', )]
    #
    # df_sel5 = pdx.groups_select(df, values='Agency_01')
    # df_sel6 = pdx.groups_select(df, values=(None, 'SKU_01'))
    # df_sel7 = pdx.groups_select(df, values=('Agency_01', 'SKU_01'))

    df = pdx.groups_select(df, values=('Agency_01', 'SKU_01'))

    X, y = pdx.xy_split(df, target=TARGET)

    X_train, X_test, y_train, y_test = pdx.train_test_split(X, y, test_size=12)

    pass


if __name__ == "__main__":
    main()
