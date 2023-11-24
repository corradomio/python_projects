import os
import logging.config
import warnings
import matplotlib.pyplot as plt
import pandasx as pdx

from sktimex.utils import plot_series
from pandasx.preprocessing import MinMaxScaler, DetrendTransformer

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


def minmax(df_all):
    methods = ['linear', 'piecewise', 'stepwise', 'global', 'poly1', 'power', 'poly3']
    # methods = ['piecewise']
    for method in methods:
        os.makedirs(f'./plots/import_kg/minmax/{method}', exist_ok=True)

        sp = 12
        dfg = pdx.groups_split(df_all)
        for g in sorted(dfg.keys()):
            # if g[0] != 'BANANA~INDIA':
            #     continue

            print(g)

            tsname = g[0].replace('/', '-')
            fname = f'./plots/import_kg/minmax/{method}/{tsname}.png'
            # if os.path.exists(fname):
            #     continue

            df = dfg[g]

            df_scaled = MinMaxScaler(method=method, columns=TARGET, sp=sp, tau=2).fit_transform(df)
            df_global = MinMaxScaler(method='global', columns=TARGET, sp=sp).fit_transform(df)

            y_scaled = df_scaled[TARGET]
            y = df_global[TARGET]

            plot_series(y, y_scaled, labels=['y', 'y scaled'], title=f"{g[0]} ({sp})")

            plt.savefig(fname)
            plt.close()

            # break
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

    minmax(df_all)

    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
