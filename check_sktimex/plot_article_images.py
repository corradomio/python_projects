#
# pos saw12 sin12 sinabs12 sq12 tri12 was12
#
import pandas as pd
import pandasx as pdx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from synth import create_synthetic_data


MODELS = [
    # "pos",
    "saw", "sin", "sinabs", "sq", "tri", "was"
]

SEASONALITIES = [
    3, 6, 12, 24, 36, 48
]


def plot_timeseries(df: pd.DataFrame):
    plt.clf()
    plt.figure(figsize=(10,5))

    dfdict = pdx.groups_split(df, groups=["cat"])

    models = [f"{m}24" for m in MODELS]
    for i, m in enumerate(models):
        data = dfdict[(m,)]
        plt.subplot(231+i)
        plt.plot(data["y"])
        plt.title(m[:-2])

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
        if i < 3:
            ax.set_xticklabels([])
        ax.set_yticklabels([])
        pass

    plt.tight_layout()

    # plt.show()
    plt.savefig("plots_article/models.png", dpi=300)
    pass


def plot_seasonalities(df: pd.DataFrame, m: str="sin"):
    plt.clf()
    plt.figure(figsize=(10,5))

    dfdict = pdx.groups_split(df, groups=["cat"])

    models = [f"{m}{s}" for s in SEASONALITIES]
    for i, m in enumerate(models):
        data = dfdict[(m,)]
        plt.subplot(231 + i)
        plt.plot(data["y"])
        plt.title(m)

        ax = plt.gca()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
        if i < 3:
            ax.set_xticklabels([])

        ax.set_yticklabels([])
        pass

    plt.tight_layout()

    # plt.show()
    plt.savefig("plots_article/seasonalities.png", dpi=300)
    pass


def main():
    df = create_synthetic_data(12 * 10, 0.0, 1, 0.33)

    plot_timeseries(df)
    plot_seasonalities(df)
    pass


if __name__ == "__main__":
    main()

