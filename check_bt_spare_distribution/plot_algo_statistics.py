import matplotlib.pyplot as plt
import numpy as np
from path import Path as path

from stdlib.dictx import dict_get
from stdlib.jsonx import load

RESULTS = path("./results_sd")

ALGO_NAMES = [
    "rvhc-perm", "rvga-perm", "rvsa-perm", "rkeda-perm", "ilp-relaxed", "ilp-180"
    # "RVHC"     "RVGA"       "RVSA"       "RKEDA"       "ILPR"         "ILP"
]
ALGO_LABELS = [
    "RVHC", "RVGA", "RVSA", "RKEDA", "ILPR", "ILP"
]

ALGO_MAP = {
"rvhc-perm":"RVHC",
"rvga-perm": "RVGA",
"rvsa-perm": "RVSA",
"rkeda-perm": "RKEDA",
"ilp-relaxed": "ILPR",
"ilp-180": "ILP"
}

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


def print_algos_stats_boxplot(nw, algos_stats:dict):

    # names = [
    #     algo_stats["name"]
    #     for algo_stats in algos_stats.values()
    # ]
    names = ALGO_NAMES[:4]
    D = [
        algos_stats[name]["bestFits"]
        for name in names
    ]

    pwidth = 6
    pheight = 3.5  # 3.5

    plt.clf()
    plt.gcf().set_size_inches(pwidth, pheight)

    plt.boxplot(
        D,
        positions=[1,2,3,4],
        showmeans=True, meanline=True,
        notch=False
    )

    plt.gca().set_xticklabels(ALGO_LABELS[:4])
    # plt.ylim(7000, 12000)
    plt.ylabel("solution value")
    # plt.xlabel("algorithms")
    # plt.title(f"Algorithms statistics ({nw})")
    plt.title(f"N={nw}")
    plt.tight_layout(pad=0.5)

    fname = f"results_plots/boxplot-{nw:03}.png"
    plt.savefig(fname, dpi=300)
    print(fname)
    pass


def plot_algos_stats_all_sizes():
    for d in RESULTS.dirs():
        # print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        print_algos_stats_boxplot(nw, algos_stats)
    pass


def plot_algo_behaviour_all_sizes():
    algos_stats_dict = {}
    for d in RESULTS.dirs():
        # print(d)
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

        pwidth = 6
        pheight = 3  # 3.5

        plt.clf()
        plt.gcf().set_size_inches(pwidth, pheight)

        plt.fill_between(data[:,0], data[:, 3], data[:, 4], alpha=.5, linewidth=0)
        plt.plot(data[:,0], data[:, 1])
        plt.xlabel("number of warehouses")
        plt.ylabel("solution value")
        # plt.ylim(7500, 12000)
        plt.ylim(6000, 12000)
        plt.title(f"Algorithm {ALGO_MAP[name]}")
        plt.tight_layout(pad=0.5)

        fname = f"results_plots/line-{name}.png"
        plt.savefig(fname, dpi=300)
        print(fname)
        pass
    pass
# end


def plot_all_algos_behaviour_all_sizes():
    algos_stats_dict = {}
    for d in RESULTS.dirs():
        # print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        algos_stats_dict[nw] = algos_stats
    # end

    pwidth = 6
    pheight = 3.5  # 4

    plt.clf()
    plt.gcf().set_size_inches(pwidth, pheight)

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
    plt.xlabel("number of warehouses")
    plt.ylabel("mean solution value")
    plt.ylim(6000, 12000)
    plt.legend(ALGO_LABELS)
    plt.tight_layout(pad=0.5)

    fname = f"results_plots/scatter-all.png"
    plt.savefig(fname, dpi=300)
    print(fname)
    pass


def main():
    plot_algos_stats_all_sizes()
    plot_algo_behaviour_all_sizes()
    plot_all_algos_behaviour_all_sizes()
    pass


if __name__ == "__main__":
    main()
