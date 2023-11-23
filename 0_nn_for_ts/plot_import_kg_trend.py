from joblibx import Parallel, delayed
import logging.config
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pandasx as pdx
from sktimex.utils import plot_series
from pandasx.preprocessing.minmax import *

# hide warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("root")

DATA_DIR = "./data"
PLOTS_DIR = "D:/Projects.github/python_projects/check_timeseries_nn/plots"

DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
GROUPS = ['item_country']


MINMAX_METHODS = ['identity', 'linear', 'piecewise', 'stepwise', 'global', 'poly1', 'power', 0.8]
TREND_METHODS  = ['identity', 'linear', 'piecewise', 'stepwise', 'global', 'poly1', 'power']


def plot_trend_method(method, df_all):
    print('---')
    print('---', method)
    print('---')
    os.makedirs(f'./plots/import_kg/trend/{method}', exist_ok=True)

    tsid = 0
    sp = 12
    dfg = pdx.groups_split(df_all)
    for g in sorted(dfg.keys()):
        print(g)

        tsid += 1
        tsname = g[0].replace('/', '-')
        fname = f'./plots/import_kg/trend/{method}/{tsname}.png'
        # if os.path.exists(fname):
        #     continue

        try:
            df = dfg[g].reset_index()
            ys = df[TARGET]

            y = ys.to_numpy()
            x = np.arange(len(y))

            if method == 'linear':
                trend_fun = fit_function(poly1, x, y)
                t = trend_fun(x)
            elif method == 'poly1':
                trend_fun = fit_function(poly1, x, y)
                t = trend_fun(x)
            elif method == 'poly3':
                trend_fun = fit_function(poly3, x, y)
                t = trend_fun(x)
            elif method == 'poly5':
                trend_fun = fit_function(poly5, x, y)
                t = trend_fun(x)
            elif method == 'power':
                trend_fun = fit_function(power1, x, y)
                t = trend_fun(x)
            elif method == 'exp':
                trend_fun = fit_function(exp1, x, y)
                t = trend_fun(x)
            elif method == 'piecewise':
                xy = select_seasonal_values(sp, x, y, centered=True)
                t = interpolate_bounds(xy, x)
            elif method == 'stepwise':
                xy = select_seasonal_values(sp, x, y, centered=False)
                t = select_bounds(xy, x)
            else:
                raise ValueError(f'Unsupported method {method}')

            ts = pd.Series(data=t, index=ys.index)

            ys.name = 'target'
            ts.name = 'trend'

            plot_series(ys, ts, labels=['y', 'trend'], title=f"TS-{tsid} ({method})")
            plt.tight_layout()

            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(e)
            pass


def plot_trend(df_all):
    methods = ['linear', 'piecewise', 'stepwise', 'poly1', 'poly3', 'poly5', 'power', 'exp']
    # methods = ['piecewise', 'stepwise']

    Parallel(n_jobs=6)(delayed(plot_trend_method)(method, df_all) for method in methods)

    # for method in methods:
    #     plot_trend_method(method, df_all)
    # end


def main():
    os.makedirs('./plots/import_kg/trend', exist_ok=True)

    df_all = pdx.read_data(
        f"{DATA_DIR}/vw_food_import_kg_train_test.csv",
        datetime=DATETIME,
        ignore=GROUPS + DATETIME[0:1] + [
            "imp_month",
            "prod_kg",
            "avg_retail_price_src_country",
            "producer_price_tonne_src_country",
            "min_temperature",
            "max_temperature",

            # "crude_oil_price",
            # "sandp_500_us",
            # "sandp_sensex_india",
            # "shenzhen_index_china",
            # "nikkei_225_japan",

            # "mean_temperature",
            "vap_pressure",
            "evaporation",
            "rainy_days",
        ],
        onehot=["imp_month"],
        dropna=True,
        na_values=['(null)'],
        index=GROUPS + DATETIME[0:1]
    )

    plot_trend(df_all)
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
