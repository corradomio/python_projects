import logging.config
import os
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
TARGET = "import_aed"
GROUPS = ['item_country']


def detrend(df_all):
    for method in ['stepwise', 'piecewise', 'linear', 'power']:
        os.makedirs(f'./plots/import_aed/detrend/{method}', exist_ok=True)

        sp = 12
        dfg = pdx.groups_split(df_all)
        for g in dfg:
            print(g)
            df = dfg[g]

            df_scaled = DetrendTransformer(method=method,
                                           method_method='median',
                                           columns=TARGET, sp=sp).fit_transform(df)
            df_global = DetrendTransformer(method='global', columns=TARGET, sp=sp).fit_transform(df)

            y_scaled = df_scaled[TARGET]
            y = df_global[TARGET]

            plot_series(y, y_scaled, labels=['y', 'y scaled'], title=f"{g[0]} ({sp})")

            tsname = g[0].replace('/', '-')
            fname = f'./plots/import_aed/detrend/{method}/{tsname}.png'
            plt.savefig(fname)
            plt.close()
        # end
    # end


def minmax(df_all):
    for method in ['stepwise', 'piecewise']:
        os.makedirs(f'./plots/import_aed/minmax/{method}', exist_ok=True)

        sp = 12
        dfg = pdx.groups_split(df_all)
        for g in dfg:
            print(g)
            df = dfg[g]

            df_scaled = MinMaxScaler(method=method, columns=TARGET, sp=sp).fit_transform(df)
            df_global = MinMaxScaler(method='global', columns=TARGET, sp=sp).fit_transform(df)

            y_scaled = df_scaled[TARGET]
            y = df_global[TARGET]

            plot_series(y, y_scaled, labels=['y', 'y scaled'], title=f"{g[0]} ({method}: {sp})")

            tsname = g[0].replace('/', '-')
            fname = f'./plots/import_aed/minmax/{tsname}.png'
            plt.savefig(fname)
            plt.close()
        # end


def main():
    os.makedirs('./plots/import_aed/minmax', exist_ok=True)
    os.makedirs('./plots/import_aed/detrend', exist_ok=True)

    df_all = pdx.read_data(
        f"{DATA_DIR}/vw_food_import_aed_train_test.csv",
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

    detrend(df_all)
    minmax(df_all)

    pass


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.config.fileConfig('logging_config.ini')
    log = logging.getLogger("root")
    log.info("Logging system configured")
    main()
