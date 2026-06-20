import matplotlib.pyplot as plt
import numpy as np
from path import Path as path
from pymoo.core import plot

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


def plot_algos_stats_boxplot(all_stats: dict[int, dict]):

    fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(7, 3))
    c = -1
    for nw in NW_LIST:
        c += 1
        ax = axs[c//3, c%3]
        twinx = ax
        # twinx = ax.twinx()
        # plt.sca(ax)

        algos_stats = all_stats[nw]

        names = ALGO_NAMES[:4]
        D = [
            algos_stats[name]["bestFits"]
            for name in names
        ]
        D = D[::-1]

        A_names = ALGO_LABELS[:4]
        A_names = A_names[::-1]

        twinx.boxplot(
            D,
            positions=[1,2,3,4],
            showmeans=True, meanline=True,
            notch=False,
            vert=False,
            tick_labels=A_names
        )

        if c%3 > 0:
            twinx.set_yticklabels([])

        twinx.text(0.75, 0.9, f"N={nw}",fontsize=8, transform=twinx.transAxes)

        # plt.gca().set_xticklabels(ALGO_LABELS[:4])
        # plt.ylim(7000, 12000)
        # plt.ylabel("solution value")
        # plt.xlabel("algorithms")
        # plt.title(f"Algorithms statistics ({nw})")
        # plt.title(f"N={nw}")

    # axs[0,0].set_yticklabels(ALGO_LABELS[:4])
    # axs[1,0].set_yticklabels(ALGO_LABELS[:4])

    plt.tight_layout(pad=0.5)

    fname = f"results_boxplot/boxplot-all.png"
    plt.savefig(fname, dpi=300)
    print(fname)
    pass


def plot_algos_boxplot_all_sizes():
    all_stats = {}
    for d in RESULTS.dirs():
        # print(d)
        nw = int(d.stem)

        algos_stats = collect_algos_stats(d)

        all_stats[nw] = algos_stats
    # end
    plot_algos_stats_boxplot(all_stats)
    pass
# end



def main():
    plot_algos_boxplot_all_sizes()
    pass


if __name__ == "__main__":
    main()
