import logging.config
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

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


def plot_sp(sp, x):
    n = len(x)//sp
    for i in range(0, n+1):
        plt.axvline(i*sp)


def plot_bounds(df_all):
    methods = ['linear', 'poly1', 'poly3', 'poly5', 'power', 'exp']
    methods = ['exp']

    for method in methods:
        print("---")
        print("---", method)
        print("---")
        plot_dir = f'./plots/import_kg/bounds/{method}'
        os.makedirs(plot_dir, exist_ok=True)

        sp = 12
        tsid = 0

        dfg = pdx.groups_split(df_all)
        for g in sorted(dfg.keys()):
            try:
                print(g)

                tsid += 1
                tsname = g[0].replace('/', '-')
                fname = f'{plot_dir}/{tsname}.png'
                if os.path.exists(fname):
                    continue

                df = dfg[g].reset_index()
                ys = df[TARGET]
                ys.name = 'target'

                n = len(ys)
                y = ys.to_numpy()
                x = np.arange(n)

                lb, ub = compute_bounds(sp, x, y)
                if method == 'linear':
                    lbf = fit_linear_bound(lb[:, 0], lb[:, 1], upper=False)
                    ubf = fit_linear_bound(ub[:, 0], ub[:, 1], upper=True)
                elif method == 'poly1':
                    lbf = fit_bound(poly1, lb[:, 0], lb[:, 1], upper=False)
                    ubf = fit_bound(poly1, ub[:, 0], ub[:, 1], upper=True)
                elif method == 'poly3':
                    lbf = fit_bound(poly3, lb[:, 0], lb[:, 1], upper=False)
                    ubf = fit_bound(poly3, ub[:, 0], ub[:, 1], upper=True)
                elif method == 'poly5':
                    lbf = fit_bound(poly5, lb[:, 0], lb[:, 1], upper=False)
                    ubf = fit_bound(poly5, ub[:, 0], ub[:, 1], upper=True)
                elif method == 'power':
                    lbf = fit_bound(power1, lb[:, 0], lb[:, 1], upper=False)
                    ubf = fit_bound(power1, ub[:, 0], ub[:, 1], upper=True)
                elif method == 'exp':
                    lbf = fit_bound(exp1, lb[:, 0], lb[:, 1], upper=False)
                    ubf = fit_bound(exp1, ub[:, 0], ub[:, 1], upper=True)
                else:
                    raise ValueError(f'Unsupported {method}')

                lby = lbf(x)
                uby = ubf(x)

                # plt.title(f"TS-{tsid}")
                plot_series(ys, labels=['target'], title=f"TS-{tsid}")

                # bounds
                plt.plot(lby, c='g')
                plt.plot(uby, c='r')

                # reference points
                plt.scatter(lb[:, 0], lb[:, 1], c='g', s=100)
                plt.scatter(ub[:, 0], ub[:, 1], c='r', s=100)

                # periods
                plot_sp(sp, x)

                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
            except Exception as e:
                print(e)
                pass
        # end
    # end
# end


def main():
    os.makedirs('./plots/import_kg/minmax', exist_ok=True)
    os.makedirs('./plots/import_kg/detrend', exist_ok=True)

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

    df_all = pdx.preprocessing.OutlierTransformer(columns=TARGET, sp=12, outlier_std=3).fit_transform(df_all)

    plot_bounds(df_all)

    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
