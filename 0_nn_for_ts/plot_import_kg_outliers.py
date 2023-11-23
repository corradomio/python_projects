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


def plot_points(y, idiff, ax):
    xp = idiff
    yp = y[idiff]

    ax.scatter(xp, yp, c='r', s=50, linewidth=3)
    pass


def plot_outliers(df_all):

    m_glob = 'mean'
    m_seas = 'mean'

    plot_dir = f'./plots/import_kg/outliers/{m_glob}-{m_seas}'
    os.makedirs(plot_dir, exist_ok=True)

    tsid = 0
    sp = 12
    dfg = pdx.groups_split(df_all)
    for g in sorted(dfg.keys()):
        print(g)

        tsid += 1
        tsname = g[0].replace('/', '-')
        fname = f'{plot_dir}/{tsname}.png'
        # if os.path.exists(fname):
        #     continue

        fig, axs = plt.subplots(3, figsize=plt.figaspect(0.25))
        # fig.suptitle(f"TS-{tsid}")
        fig.suptitle(f"TS-{tsid} (g={m_glob}, s={m_seas})")

        try:
            df = dfg[g].reset_index()

            df_glob = pdx.OutlierTransformer(columns=TARGET, sp=None, outlier_std=2, strategy=m_glob).fit_transform(df)
            df_seas = pdx.OutlierTransformer(columns=TARGET, sp=sp, outlier_std=2, strategy=m_seas).fit_transform(df)

            y = df[TARGET]
            y_glob = df_glob[TARGET]
            y_seas = df_seas[TARGET]

            ymin, ymax = y.min(), y.max()
            ydelta = (ymax - ymin)
            ymin -= 0.1*ydelta
            ymax += 0.1*ydelta

            y.name = 'target'
            y_glob.name = 'target'
            y_seas.name = 'target'

            plot_series(y, labels=['target'], ax=axs[0])
            plot_series(y_glob, labels=['global'], ax=axs[1])
            plot_series(y_seas, labels=['seasonal'], ax=axs[2])

            n = len(y)
            d_glob = []
            d_seas = []
            for i in range(n):
                if y.iloc[i] != y_glob.iloc[i]:
                    d_glob.append(i)
                if y.iloc[i] != y_seas.iloc[i]:
                    d_seas.append(i)

            for ax in axs:
                ax.set_ylim((ymin, ymax))
                ax.set_ylabel(None)
                ax.set_xlabel(None)
                ax.set_yticklabels([])
                ax.set_xticklabels([])

            plot_points(y_glob, d_glob, ax=axs[1])
            plot_points(y_seas, d_seas, ax=axs[2])

            # plot_series(y, y_glob, y_seas, labels=['target', 'global', 'seasonal'], title=f"TS-{tsid})")
            plt.tight_layout()

            plt.savefig(fname)
            plt.close()

            # break
        except Exception as e:
            print(e)
            pass


def main():
    os.makedirs('./plots/import_kg/outliers', exist_ok=True)

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

    plot_outliers(df_all)
    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
