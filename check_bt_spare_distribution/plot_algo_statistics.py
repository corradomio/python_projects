import matplotlib.pyplot as plt
import numpy as np
from path import Path as path

from stdlib.dictx import dict_get
from stdlib.jsonx import load

RESULTS = path("./results_sd")

ALGO_NAMES = [
    "rvhc-perm", "rvga-perm", "rvsa-perm", "rkeda-perm", "ilp-relaxed", "ilp-180"
    # "RVHC", "RVGA", "RVSA", "RKEDA", "ILPR"
]
ALGO_LABELS = [
    "RVHC", "RVGA", "RVSA", "RKEDA", "ILPR", "ILP"
]

NW_LIST = [50,60,70,80,90,100]


def compute_algo_statistics(f: path) -> dict:
    print(f)
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
    names = ALGO_NAMES[:4]
    D = [
        algos_stats[name]["bestFits"]
        for name in names
    ]

    plt.clf()

    plt.boxplot(
        D,
        positions=[1,2,3,4],
        showmeans=True, meanline=True,
        notch=False
    )

    plt.gca().set_xticklabels(names)
    # plt.ylim(7000, 12000)
    plt.ylabel("fitness value")
    plt.xlabel("algorithms")
    plt.title(f"Algorithms stats ({nw})")

    fname = f"results_plots/boxplot-{nw:03}.png"
    plt.savefig(fname, dpi=300)

    # plt.show()
    pass


def plot_algos_stats_by_size():
    for d in RESULTS.dirs():
        print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        print_algos_stats(nw, algos_stats)
    pass


def plot_algo_behaviour_all_sizes():
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
        plt.ylabel("fitness value")
        # plt.ylim(7500, 12000)
        plt.ylim(6000, 12000)
        plt.title(f"Algorithm {name}")
        plt.tight_layout()

        fname = f"results_plots/line-{name}.png"
        # plt.show()
        plt.savefig(fname, dpi=300)
        pass

    # end

    pass
# end


def plot_all_algos_behaviour_all_sizes():
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
    plt.ylabel("fitness value (mean)")
    plt.ylim(6000, 12000)
    plt.title(f"Algorithms' means of best solutions")
    plt.legend(ALGO_LABELS)

    fname = f"results_plots/line-all.png"
    # plt.show()
    plt.savefig(fname, dpi=300)

    pass


def main():
    # plot_algos_stats_by_size()
    # plot_algo_behaviour_all_sizes()
    plot_all_algos_behaviour_all_sizes()
    pass


if __name__ == "__main__":
    main()
