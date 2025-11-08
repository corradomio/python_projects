import numpy as np
import pandas as pd
from path import Path as path
from stdlib.dictx import dict_get
from stdlib.jsonx import load
import matplotlib.pyplot as plt
import seaborn as sns


RESULTS = path("./results_sd")

ALGO_NAMES = [
    "rvhc-perm", "rvga-perm", "rvsa-perm", "rkeda-perm", "ilp-relaxed", "ilp-180"
]
ALGO_LABELS = [
    "RVHC", "RVGA", "RVSA", "RKEDA", "ILPR", "ILP180"
]

NW_LIST = [50,60,70,80,90,100]


def compute_algo_statistics(f: path) -> dict:
    # print(f)
    jdata = load(f)

    algo_name = dict_get(jdata, ["algoParams", "name"])
    num_centers = dict_get(jdata, ["experimentParams", "numCenters"])

    # collect best fits
    results: list[dict] = dict_get(jdata, ["results"])
    best_fits = [
        dict_get(r, ["params", "totalDistance"])
        for r in results
    ]

    return dict(
        name=algo_name,
        numCenters=num_centers,
        bestFits=best_fits
    )


def collect_algos_stats(d: path) -> dict:
    algos_stats = {}
    for f in d.files("*.json"):
        algo_stats = compute_algo_statistics(f)

        algo_name = algo_stats["name"]

        algos_stats[algo_name] = algo_stats
    return algos_stats


def print_algos_stats(nw, algos_stats:dict):

    # names = [
    #     algo_stats["name"]
    #     for algo_stats in algos_stats.values()
    # ]
    # names = ALGO_NAMES
    # D = [
    #     algos_stats[name]["bestFits"]
    #     for name in names
    # ]

    for name in ALGO_NAMES:
        bestFits = np.array(algos_stats[name]["bestFits"], dtype=float)
        print(f"{nw:3} & {name:10} & {bestFits.mean():.3f} & {bestFits.std():.3f} \\\\")

    # plt.clf()
    #
    # plt.boxplot(
    #     D,
    #     positions=[1,2,3,4],
    #     showmeans=True, meanline=True,
    #     notch=False
    # )
    #
    # plt.gca().set_xticklabels(names)
    # # plt.ylim(7000, 12000)
    # plt.ylabel("solution value")
    # plt.xlabel("algorithms")
    # plt.title(f"Algorithms stats ({nw})")
    #
    # fname = f"results_plots/boxplot-{nw:03}.png"
    # plt.savefig(fname, dpi=300)

    # plt.show()
    pass


def compute_algos_stats_by_size():
    for d in RESULTS.dirs():
        # print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        print_algos_stats(nw, algos_stats)
    pass


def compute_algo_behaviour_all_sizes():
    algos_stats_dict = {}
    for d in RESULTS.dirs():
        print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        algos_stats_dict[nw] = algos_stats
    # end

    for name in ALGO_NAMES:
        means = []
        stdvs = []
        for nw in NW_LIST:
            algo_stats = algos_stats_dict[nw][name]
            best_fits = np.array(algo_stats["bestFits"])
            means.append(best_fits.mean())
            stdvs.append(best_fits.std())
        # end

        data = np.array([
            NW_LIST,
            means,
            stdvs,
            [0]*6, # lo limit
            [0]*6, # hi limit
        ]).T

        data[:, 3] = data[:, 1] - data[:, 2]
        data[:, 4] = data[:, 1] + data[:, 2]

        # df = pd.DataFrame(data=data, columns=["nw", "mean", "std"])

        plt.clf()
        plt.fill_between(data[:,0], data[:, 3], data[:, 4], alpha=.5, linewidth=0)
        plt.plot(data[:,0], data[:, 1])
        plt.xlabel("n warehouses")
        plt.ylabel("solution value")
        plt.ylim(7000, 12000)
        plt.title(f"Algorithm {name}")
        plt.tight_layout(pad=0.5)

        fname = f"results_plots/line-{name}.png"
        plt.savefig(fname, dpi=300)
        print(fname)
        pass
    pass
# end


def compute_all_algos_behaviour_all_sizes():
    algos_stats_dict = {}
    for d in RESULTS.dirs():
        print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        algos_stats_dict[nw] = algos_stats
    # end

    plt.clf()

    for name in ALGO_NAMES:
        means = []
        stdvs = []
        for nw in NW_LIST:
            algo_stats = algos_stats_dict[nw][name]
            best_fits = np.array(algo_stats["bestFits"])
            means.append(best_fits.mean())
            stdvs.append(best_fits.std())
        # end

        data = np.array([
            NW_LIST,
            means,
            stdvs,
            [0]*6, # lo limit
            [0]*6, # hi limit
        ]).T

        data[:, 3] = data[:, 1] - data[:, 2]
        data[:, 4] = data[:, 1] + data[:, 2]

        # df = pd.DataFrame(data=data, columns=["nw", "mean", "std"])

        # plt.fill_between(data[:,0], data[:, 3], data[:, 4], alpha=.5, linewidth=0)
        plt.scatter(data[:,0], data[:, 1])
        pass
    # end
    plt.xlabel("n warehouses")
    plt.ylabel("mean solution value")
    plt.ylim(7000, 12000)
    plt.title(f"Algorithms' means of best solutions")
    plt.legend(ALGO_LABELS)
    plt.tight_layout(pad=0.5)

    fname = f"results_plots/line-all.png"
    plt.savefig(fname, dpi=300)
    print(fname)
    pass


def main():
    compute_algos_stats_by_size()
    # compute_algo_behaviour_all_sizes()
    # compute_all_algos_behaviour_all_sizes()
    pass


if __name__ == "__main__":
    main()
