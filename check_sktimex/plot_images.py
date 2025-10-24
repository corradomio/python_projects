import pandasx as pdx
import sktime.utils.plotting
import matplotlib.pyplot as plt
from joblibx import Parallel, delayed


def to_name(g: tuple) -> str:
    country, item = g
    item = item.replace("/", "_").replace(" ", "_")
    return f"{country}-{item}"


def plot_series(dfdict, g):
    print(g)
    dfg = dfdict[g]
    sktime.utils.plotting.plot_series(dfg["import_kg"], labels=["import_kg"], title=str(g))

    name = to_name(g)
    plt.savefig(f"plots/{name}.png", dpi=300)
    plt.close()


def main():
    df = pdx.read_data("data/vw_food_import_kg_train_test_area_skill.csv")
    dfdict = pdx.groups_split(df, groups=["country", "item"])
    # for g in dfdict:
    #     plot_series(dfdict, g)
    Parallel(n_jobs=6)(delayed(plot_series)(dfdict, g) for g in dfdict)
    # end


if __name__ == "__main__":
    main()
