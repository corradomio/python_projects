import logging.config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandasx as pdx
from pandasx.preprocessing.minmax import compute_bounds, fit_bound, fit_linear_bound
from pandasx.preprocessing.minmax import poly1, poly3, power1, log1


def gen_data(n=120):
    x = np.arange(0, n, 1)
    y = 20*np.log(x/5+1) + 5*np.sin(np.pi*2*x/12) + np.random.randn(n)
    return x, y


def plot_sp(sp, x):
    n = len(x)//sp
    for i in range(0, n+1):
        plt.axvline(i*sp)


def main1():
    sp = 12
    x, y = gen_data(sp*10)

    lb, ub = compute_bounds(sp, x, y)

    # plt.plot(x, y)
    # plt.scatter(lb[:, 0], lb[:, 1])
    # plt.scatter(ub[:, 0], ub[:, 1])
    # plot_sp(sp, x)
    # plt.show()

    # lbf = fit_linear_bound(lb[:, 0], lb[:, 1], upper=False)
    # ubf = fit_linear_bound(ub[:, 0], ub[:, 1], upper=True)

    # lbf = fit_bound(poly1, lb[:, 0], lb[:, 1], upper=False)
    # ubf = fit_bound(poly1, ub[:, 0], ub[:, 1], upper=True)

    lbf = fit_bound(log1, lb[:, 0], lb[:, 1], upper=False)
    ubf = fit_bound(log1, ub[:, 0], ub[:, 1], upper=True)

    plt.plot(x, y)
    lby = lbf(x)
    uby = ubf(x)
    plt.plot(x, y)
    plt.plot(x, lby)
    plt.plot(x, uby)
    plt.show()


    pass


def main2():
    df = pdx.read_data(
        "./data/airline.csv",
        datetime=('Period', '%Y-%m', 'M'),
        ignore=['Period'],
        index=['Period']
    )

    TARGET = 'Number of airline passengers'

    sp = 12
    y = df[TARGET].to_numpy()
    x = np.arange(len(y))

    lb, ub = compute_bounds(sp, x, y)

    # plt.plot(x, y)
    # plt.scatter(lb[:, 0], lb[:, 1])
    # plt.scatter(ub[:, 0], ub[:, 1])
    # plot_sp(sp, x)
    # plt.show()

    # lbf = fit_linear_bound(lb[:, 0], lb[:, 1], upper=False)
    # ubf = fit_linear_bound(ub[:, 0], ub[:, 1], upper=True)

    lbf = fit_bound(power1, lb[:, 0], lb[:, 1], upper=False)
    ubf = fit_bound(power1, ub[:, 0], ub[:, 1], upper=True)

    # plt.plot(x, y)
    # lby = lbf(x)
    # uby = ubf(x)
    # plt.plot(x, y)
    # plt.plot(x, lby)
    # plt.plot(x, uby)
    # plt.show()

    uy = ubf(x)
    ly = lbf(x)

    yt = (y - ly)/(uy - ly)

    plt.plot(x, yt)
    plt.show()

    pass


LOGGER = logging.getLogger("root")

DATA_DIR = "./data"
PLOTS_DIR = "D:/Projects.github/python_projects/check_timeseries_nn/plots"

DATETIME = ["imp_date", "[%Y/%m/%d %H:%M:%S]", "M"]
TARGET = "import_kg"
GROUPS = ['item_country']


def plot_sp(sp, x):
    n = len(x)//sp
    for i in range(0, n+1):
        plt.axvline(i*sp, c='lightgray')


def check_minmax(y, k=1.):
    sp = 12
    n = len(y)
    x = np.arange(n)

    lb, ub = compute_bounds(sp, x, y)

    # lbf = fit_linear_bound(lb[:, 0], lb[:, 1], upper=False)
    # ubf = fit_linear_bound(ub[:, 0], ub[:, 1], upper=True)

    lbf = fit_bound(poly3, lb[:, 0], lb[:, 1], k=k, upper=False)
    ubf = fit_bound(poly3, ub[:, 0], ub[:, 1], k=k, upper=True)

    lby = lbf(x)
    uby = ubf(x)

    # plt.figaspect(0.25)
    plt.gca().set_aspect(20)
    # plt.plot(x, y)
    # plt.scatter(lb[:, 0], lb[:, 1], c='g', s=25)
    # plt.scatter(ub[:, 0], ub[:, 1], c='r', s=25)

    sns.lineplot(x=x, y=y)
    sns.scatterplot(x=lb[:, 0], y=lb[:, 1], c='g', s=50)
    sns.scatterplot(x=ub[:, 0], y=ub[:, 1], c='r', s=50)

    plt.plot(x, lby, c='g')
    plt.plot(x, uby, c='r')

    plot_sp(sp, x)

    plt.tight_layout()
    plt.show()

    pass



def main():
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

    df = pdx.groups_select(df_all, values=['BANANA~ECUADOR'])
    y = df[TARGET].to_numpy()/1e7

    check_minmax(y, k=1)
    pass
# end


if __name__ == "__main__":
    main()
