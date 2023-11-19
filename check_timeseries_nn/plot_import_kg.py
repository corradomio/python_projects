from joblibx import Parallel, delayed
import logging
import os.path
import warnings
import pandasx as pdx
import matplotlib.pyplot as plt
from sktime.utils.plotting import plot_series

# hide warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger("root")

DATA_DIR = "./data"
PLOTS_DIR = "D:/Projects.github/python_projects/check_timeseries_nn/plots"

DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
GROUPS = ['item_country']


def plot(i, n, g, df):
    print(f"[{i:4}/{n}]", g)
    item = g[0].replace('/', '_')
    fname = f"{PLOTS_DIR}/{TARGET}/{item}.png"
    if os.path.exists(fname):
        return

    plot_series(df[TARGET], labels=[TARGET], title=str(g))

    plt.savefig(fname, dpi=300)
    plt.tight_layout()
    plt.close()


def main():
    df_all = pdx.read_data(f"{DATA_DIR}/vw_food_import_train_test_newfeatures.csv",
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

    dfg = pdx.groups_split(df_all)
    n = len(dfg)

    Parallel(n_jobs=12)(delayed(plot)(i, n, g, dfg[g]) for i, g in enumerate(dfg.keys()))

    # for g in dfg:
    #     print(g)
    #     df = dfg[g]
    #
    #     plot_series(df[TARGET], labels=[TARGET], title=str(g))
    #
    #     item = g[0].replace('/', '_')
    #     fname = f"./plots/import_kg/{item}.png"
    #     plt.savefig(fname, dpi=300)
    #     plt.tight_layout()
    #     plt.close()
    pass



if __name__ == "__main__":
    main()
